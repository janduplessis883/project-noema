"""
Ollama Python Wrapper - A comprehensive wrapper for interacting with Ollama locally hosted models.

This module provides a simple yet powerful interface for:
- Generating text completions
- Chat conversations
- Model management (list, pull, delete, show)
- Streaming responses
- Error handling and connection management

Author: AI Assistant
Date: 2025-10-22
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Union, Generator, Any
from dataclasses import dataclass
from enum import Enum


class OllamaError(Exception):
    """Custom exception for Ollama-related errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Exception raised when connection to Ollama server fails."""
    pass


class OllamaModelError(OllamaError):
    """Exception raised when model-related errors occur."""
    pass


@dataclass
class OllamaResponse:
    """Standard response wrapper for Ollama API responses."""
    content: str
    model: str
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    raw_response: Optional[Dict] = None


class OllamaWrapper:
    """
    A comprehensive wrapper for interacting with Ollama locally hosted models.
    
    This class provides methods for:
    - Text generation and chat completions
    - Model management operations
    - Streaming responses
    - Connection pooling and error handling
    
    Example:
        >>> ollama = OllamaWrapper()
        >>> response = ollama.generate("llama3.2", "Why is the sky blue?")
        >>> print(response.content)
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize the Ollama wrapper.
        
        Args:
            base_url: Base URL for the Ollama server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/version", 
                                      timeout=self.timeout)
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Status code: {response.status_code}"
                )
            self.logger.info(f"Successfully connected to Ollama server at {self.base_url}")
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Could not connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running. Error: {str(e)}"
            )
    
    def _make_request(self, endpoint: str, data: Dict[str, Any], 
                     stream: bool = False) -> Union[Dict, Generator]:
        """
        Make HTTP request to Ollama API with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Request payload
            stream: Whether to stream the response
            
        Returns:
            Response data or generator for streaming
            
        Raises:
            OllamaError: If request fails after all retries
        """
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    response = self.session.post(url, json=data, stream=True,
                                               timeout=self.timeout)
                    if response.status_code == 200:
                        return self._handle_streaming_response(response)
                else:
                    response = self.session.post(url, json=data,
                                               timeout=self.timeout)
                    if response.status_code == 200:
                        return response.json()
                
                # Handle HTTP errors
                if response.status_code == 404:
                    error_msg = response.json().get('error', 'Model not found')
                    raise OllamaModelError(f"Model error: {error_msg}")
                elif response.status_code == 400:
                    error_msg = response.json().get('error', 'Bad request')
                    raise OllamaError(f"Bad request: {error_msg}")
                else:
                    response.raise_for_status()
                    
            except (requests.exceptions.RequestException, OllamaError) as e:
                if attempt == self.max_retries:
                    raise OllamaError(f"Request failed after {self.max_retries + 1} attempts: {str(e)}")
                
                self.logger.warning(f"Request attempt {attempt + 1} failed, retrying...")
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
    
    def _handle_streaming_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Handle streaming response from Ollama API."""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
                    elif 'response' in data:
                        yield data['response']
                except json.JSONDecodeError:
                    continue
    
    def generate(self, 
                model: str, 
                prompt: str,
                system: Optional[str] = None,
                stream: bool = False,
                format: Optional[str] = None,
                options: Optional[Dict[str, Any]] = None,
                keep_alive: Optional[str] = None) -> Union[OllamaResponse, Generator[str, None, None]]:
        """
        Generate text completion using the specified model.
        
        Args:
            model: Model name (e.g., 'llama3.2', 'mistral')
            prompt: Input prompt for text generation
            system: System prompt to set context
            stream: Whether to stream the response
            format: Response format ('json' for structured output)
            options: Additional model parameters (temperature, top_p, etc.)
            keep_alive: How long to keep model loaded (e.g., '5m', '1h')
            
        Returns:
            OllamaResponse or generator for streaming
            
        Example:
            >>> response = ollama.generate("llama3.2", "Explain quantum computing")
            >>> print(response.content)
        """
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': stream
        }
        
        if system:
            payload['system'] = system
        if format:
            payload['format'] = format
        if options:
            payload['options'] = options
        if keep_alive:
            payload['keep_alive'] = keep_alive
        
        if stream:
            return self._make_request('/api/generate', payload, stream=True)
        else:
            response_data = self._make_request('/api/generate', payload)
            return OllamaResponse(
                content=response_data.get('response', ''),
                model=response_data.get('model', model),
                done=response_data.get('done', True),
                total_duration=response_data.get('total_duration'),
                load_duration=response_data.get('load_duration'),
                prompt_eval_count=response_data.get('prompt_eval_count'),
                prompt_eval_duration=response_data.get('prompt_eval_duration'),
                eval_count=response_data.get('eval_count'),
                eval_duration=response_data.get('eval_duration'),
                raw_response=response_data
            )
    
    def chat(self, 
             model: str, 
             messages: List[Dict[str, str]],
             stream: bool = False,
             format: Optional[str] = None,
             options: Optional[Dict[str, Any]] = None,
             keep_alive: Optional[str] = None) -> Union[OllamaResponse, Generator[str, None, None]]:
        """
        Generate chat completion with conversation history.
        
        Args:
            model: Model name
            messages: List of message dictionaries with 'role' and 'content' keys
            stream: Whether to stream the response
            format: Response format ('json' for structured output)
            options: Additional model parameters
            keep_alive: How long to keep model loaded
            
        Returns:
            OllamaResponse or generator for streaming
            
        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "How are you?"}
            ... ]
            >>> response = ollama.chat("llama3.2", messages)
            >>> print(response.content)
        """
        payload = {
            'model': model,
            'messages': messages,
            'stream': stream
        }
        
        if format:
            payload['format'] = format
        if options:
            payload['options'] = options
        if keep_alive:
            payload['keep_alive'] = keep_alive
        
        if stream:
            return self._make_request('/api/chat', payload, stream=True)
        else:
            response_data = self._make_request('/api/chat', payload)
            return OllamaResponse(
                content=response_data.get('message', {}).get('content', ''),
                model=response_data.get('model', model),
                done=response_data.get('done', True),
                total_duration=response_data.get('total_duration'),
                load_duration=response_data.get('load_duration'),
                prompt_eval_count=response_data.get('prompt_eval_count'),
                prompt_eval_duration=response_data.get('prompt_eval_duration'),
                eval_count=response_data.get('eval_count'),
                eval_duration=response_data.get('eval_duration'),
                raw_response=response_data
            )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available local models.
        
        Returns:
            List of model information dictionaries
            
        Example:
            >>> models = ollama.list_models()
            >>> for model in models:
            ...     print(f"Model: {model['name']}")
        """
        response = self._make_request('/api/tags', {})
        return response.get('models', [])
    
    def show_model(self, model: str) -> Dict[str, Any]:
        """
        Show detailed information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information dictionary
            
        Example:
            >>> info = ollama.show_model("llama3.2")
            >>> print(f"Model size: {info.get('size', 'Unknown')}")
        """
        return self._make_request('/api/show', {'model': model})
    
    def pull_model(self, model: str) -> Dict[str, Any]:
        """
        Download a model from the Ollama library.
        
        Args:
            model: Model name to pull
            
        Returns:
            Response from the pull operation
            
        Example:
            >>> ollama.pull_model("mistral")
        """
        return self._make_request('/api/pull', {'model': model})
    
    def delete_model(self, model: str) -> Dict[str, Any]:
        """
        Delete a local model.
        
        Args:
            model: Model name to delete
            
        Returns:
            Response from the delete operation
            
        Example:
            >>> ollama.delete_model("old-model")
        """
        return self._make_request('/api/delete', {'model': model})
    
    def create_model(self, 
                     model: str, 
                     modelfile: str,
                     stream: bool = False) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Create a new model from a Modelfile.
        
        Args:
            model: Name for the new model
            modelfile: Modelfile content as string
            stream: Whether to stream the response
            
        Returns:
            Response or generator for streaming
            
        Example:
            >>> modelfile = '''
            ... FROM llama3.2
            ... SYSTEM You are a helpful coding assistant.
            ... '''
            >>> ollama.create_model("coding-assistant", modelfile)
        """
        payload = {
            'model': model,
            'modelfile': modelfile,
            'stream': stream
        }
        
        if stream:
            return self._make_request('/api/create', payload, stream=True)
        else:
            return self._make_request('/api/create', payload)
    
    def generate_embeddings(self, 
                           model: str, 
                           prompt: str,
                           options: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Generate embeddings for a given prompt.
        
        Args:
            model: Model name for embeddings
            prompt: Text to generate embeddings for
            options: Additional model parameters
            
        Returns:
            List of embedding values
            
        Example:
            >>> embeddings = ollama.generate_embeddings("nomic-embed-text", "Hello world")
            >>> print(f"Embedding dimensions: {len(embeddings)}")
        """
        payload = {
            'model': model,
            'prompt': prompt
        }
        
        if options:
            payload['options'] = options
        
        response = self._make_request('/api/embeddings', payload)
        return response.get('embedding', [])
    
    def get_version(self) -> str:
        """
        Get the Ollama server version.
        
        Returns:
            Version string
            
        Example:
            >>> version = ollama.get_version()
            >>> print(f"Ollama version: {version}")
        """
        response = self.session.get(f"{self.base_url}/api/version", timeout=self.timeout)
        return response.json().get('version', 'Unknown')
    
    def get_running_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently running models.
        
        Returns:
            List of running model information
            
        Example:
            >>> running = ollama.get_running_models()
            >>> for model in running:
            ...     print(f"Running: {model['name']}")
        """
        response = self._make_request('/api/ps', {})
        return response.get('models', [])


# Convenience functions for quick usage
def generate_response(model: str, prompt: str, **kwargs) -> str:
    """
    Quick function to generate a text response.
    
    Args:
        model: Model name
        prompt: Input prompt
        **kwargs: Additional arguments passed to generate()
        
    Returns:
        Generated text response
        
    Example:
        >>> response = generate_response("llama3.2", "Explain AI")
        >>> print(response)
    """
    ollama = OllamaWrapper()
    response = ollama.generate(model, prompt, **kwargs)
    return response.content if isinstance(response, OllamaResponse) else ''.join(response)


def chat_response(model: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Quick function to get a chat response.
    
    Args:
        model: Model name
        messages: List of chat messages
        **kwargs: Additional arguments passed to chat()
        
    Returns:
        Chat response text
        
    Example:
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = chat_response("llama3.2", messages)
        >>> print(response)
    """
    ollama = OllamaWrapper()
    response = ollama.chat(model, messages, **kwargs)
    return response.content if isinstance(response, OllamaResponse) else ''.join(response)


if __name__ == "__main__":
    # Example usage
    print("Ollama Python Wrapper - Example Usage")
    print("=" * 50)
    
    try:
        # Initialize wrapper
        ollama = OllamaWrapper()
        
        # Get version
        version = ollama.get_version()
        print(f"Connected to Ollama version: {version}")
        
        # List available models
        models = ollama.list_models()
        print(f"\nAvailable models: {len(models)}")
        for model in models[:5]:  # Show first 5 models
            print(f"  - {model.get('name', 'Unknown')}")
        
        # Example text generation (uncomment to test with actual models)
        # response = ollama.generate("llama3.2", "Explain artificial intelligence in simple terms.")
        # print(f"\nGenerated response: {response.content}")
        
        print("\nWrapper initialized successfully!")
        
    except OllamaError as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running locally (ollama serve)")