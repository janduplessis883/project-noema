"""
Theme Count Function
Counts distinct semantic themes in text based on noun chunk similarity.
Requires: pip install spacy sentence-transformers scikit-learn numpy
          python -m spacy download en_core_web_sm
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import Union
import pandas as pd

def theme_count(text: str, nlp, embedder, min_dist: float = 0.35) -> int:
    """
    Estimate the number of distinct themes in a text.
    Args:
        text: Input text to analyze
        nlp: spaCy language model (e.g., en_core_web_sm)
        embedder: SentenceTransformer model for embeddings
        min_dist: Minimum semantic distance (1 - similarity) to count as different themes
    Returns:
        Integer estimate of theme count (>= 1)
    """
    # Extract content units (noun chunks); fallback to nouns if no chunks
    doc = nlp(text)
    chunks = [nc.text for nc in doc.noun_chunks]

    # Fallback: individual nouns if no noun chunks found
    if not chunks:
        chunks = [t.lemma_ for t in doc if t.pos_ == "NOUN" and not t.is_stop]

    # Single theme if insufficient content units
    if len(chunks) < 2:
        return 1

    # Generate embeddings and compute similarity matrix
    embs = embedder.encode(chunks, normalize_embeddings=True)
    sim = cosine_similarity(embs)

    # Count pairs that are sufficiently different
    # Condition: 1 - sim >= min_dist <=> sim <= (1 - min_dist)
    different_pairs = np.sum(np.triu((sim <= (1 - min_dist)), k=1))

    # Heuristic: theme count based on ratio of different pairs to total chunks
    tc = max(1, int(1 + different_pairs / max(1, len(chunks))))
    return tc


# --- Self-contained test block ---
if __name__ == "__main__":
    # Model initialization (downloads on first run)
    print("Loading models...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model... Run: python -m spacy download en_core_web_sm")
        exit(1)

    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SentenceTransformer: {e}")
        exit(1)

    # Test cases

    data = pd.read_csv('fft.csv')  # Assuming a CSV file with a column 'text' for test cases

    data.dropna(subset=['text'], inplace=True)
    text = data['text'].tolist()

    test_texts = text[13600:14200]

    # Run tests
    print("\n" + "="*60)
    print("THEME COUNT ANALYSIS")
    print("="*60)
    for i, text in enumerate(test_texts, 1):
        count = theme_count(text, nlp, embedder, min_dist=0.35)
        print(f"\nTest {i}: '{text[:100]}...'")
        print(f"  Estimated themes: {count}")
