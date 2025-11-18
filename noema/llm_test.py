from noema.ollama_wrapper import OllamaWrapper

ollama = OllamaWrapper()
response = ollama.generate("qwen3:8b", "Explain AI")
print(response.content)
