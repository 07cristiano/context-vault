# Hybrid retrieval

Keyword search is precise for names and exact phrases, while semantic vector search can find paraphrases. ContextVault combines their rank positions with Reciprocal Rank Fusion. Because the collection contains at most 300 chunks, exact NumPy cosine similarity is simpler and sufficiently fast; an approximate nearest-neighbor index is unnecessary.
