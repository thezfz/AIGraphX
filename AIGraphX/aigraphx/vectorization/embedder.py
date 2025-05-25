import logging
from typing import List, Optional, Sequence, cast
import numpy as np
from sentence_transformers import SentenceTransformer
import os
# Remove dotenv loading, no longer needed
# from dotenv import load_dotenv

# Import the settings object instead of the old config module
from aigraphx.core.config import settings

# Remove dotenv loading logic
# dotenv_path = os.path.join(os.path.dirname(__file__), \'..\', \'..\', \'.env\')
# load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Handles loading the embedding model and generating text embeddings."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initializes the embedder, loading the specified Sentence Transformer model.

        Args:
            model_name: The name of the Sentence Transformer model to load.
                        Defaults to the value from settings.sentence_transformer_model.
            device: The device to run the model on ('cpu', 'cuda', 'mps').
                    Defaults to settings.embedder_device or None (letting sentence-transformers decide).
        """
        # Use settings object for defaults, remove os.getenv calls
        self.model_name = model_name or settings.sentence_transformer_model
        self.device = device or settings.embedder_device

        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads the Sentence Transformer model."""
        if self.model is None:
            logger.info(
                f"Loading Sentence Transformer model: {self.model_name} onto device: {self.device or 'auto'} from cache: {settings.sentence_transformers_home}"
            )
            try:
                # trust_remote_code=True might be needed for some newer models from HF hub
                self.model = SentenceTransformer(
                    self.model_name, 
                    device=self.device, 
                    cache_folder=settings.sentence_transformers_home,
                    trust_remote_code=True
                )
                logger.info(f"Model {self.model_name} loaded successfully.")
                # You can log the embedding dimension if needed
                # logger.info(f"Model embedding dimension: {self.get_embedding_dimension()}")
            except Exception as e:
                logger.exception(
                    f"Failed to load Sentence Transformer model '{self.model_name}': {e}"
                )
                # Depending on requirements, you might want to raise the exception
                # raise

    def get_embedding_dimension(self) -> int:
        """Returns the embedding dimension of the loaded model."""
        if self.model is None:
            logger.error("Embedding model is not loaded, cannot get dimension.")
            return 0  # Return 0 if model is not loaded
        try:
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                logger.error("Model returned None for embedding dimension.")
                return 0  # Return 0 if model gives None dimension
            return int(dimension) # Explicitly cast to int
        except Exception as e:
            logger.exception(f"Error getting embedding dimension: {e}")
            return 0  # Return 0 on unexpected error

    def embed(self, text: Optional[str]) -> Optional[np.ndarray]:
        """Embeds a single text string.

        Args:
            text: The text string to embed. Accepts None.

        Returns:
            A numpy array representing the embedding, or None if embedding fails
            or input is invalid.
        """
        if self.model is None:
            logger.warning("Embedder model not loaded. Cannot embed.")
            return None
        # Add check for invalid input
        if not text or not isinstance(text, str):
            logger.debug(f"Invalid input text for embedding: {text}")
            return None

        try:
            # Ensure model is not None before accessing encode
            embedding = self.model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return cast(Optional[np.ndarray], embedding)
        except Exception as e:
            logger.exception(f"Error embedding text '{text[:50]}...': {e}")
            return None

    def embed_batch(self, texts: Sequence[Optional[str]]) -> Optional[np.ndarray]:
        """Embeds a batch of text strings.

        Args:
            texts: A sequence of strings to embed. Accepts None values in the list.

        Returns:
            A numpy array where each row is an embedding, or None if embedding fails
            or the input list is effectively empty.
        """
        if self.model is None:
            logger.warning("Embedder model not loaded. Cannot embed batch.")
            return None

        # Filter out None or non-string items from the input list
        valid_texts = [t for t in texts if t and isinstance(t, str)]

        if not valid_texts:
            logger.debug("Input text list is empty or contains no valid strings.")
            # Return empty array with correct shape if input is empty
            dim = self.get_embedding_dimension()
            if dim > 0:
                return np.empty((0, dim), dtype=np.float32)
            else:
                return None  # Cannot determine shape if model failed loading

        try:
            # Determine if progress bar should be shown
            show_progress_bar = len(valid_texts) > 50  # Example threshold
            # Ensure model is not None before accessing encode
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            )
            return cast(Optional[np.ndarray], embeddings)
        except Exception as e:
            logger.exception(
                f"Error embedding batch starting with '{valid_texts[0][:50]}...': {e}"
            )
            return None


# Example usage (for testing)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     embedder = TextEmbedder() # Uses default model from env or fallback
#     if embedder.model:
#         print(f"Model Dim: {embedder.get_embedding_dimension()}")
#         text1 = "This is the first test sentence."
#         text2 = "This is a second, different sentence."
#         text3 = "First test sentence is this one."
#         invalid_text = ""

#         emb1 = embedder.embed(text1)
#         emb2 = embedder.embed(text2)
#         emb3 = embedder.embed(text3)
#         emb_invalid = embedder.embed(invalid_text)

#         print("Embedding 1:", emb1[:5], "... Shape:", emb1.shape if emb1 is not None else None)
#         print("Embedding 2:", emb2[:5], "... Shape:", emb2.shape if emb2 is not None else None)
#         print("Embedding 3:", emb3[:5], "... Shape:", emb3.shape if emb3 is not None else None)
#         print("Embedding Invalid:", emb_invalid)

#         # Calculate cosine similarity (requires numpy)
#         if emb1 is not None and emb3 is not None:
#             similarity = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
#             # Since we use normalize_embeddings=True, dot product is enough
#             similarity_dot = np.dot(emb1, emb3)
#             print(f"Similarity (1 vs 3) - Dot Product: {similarity_dot:.4f}")

#         if emb1 is not None and emb2 is not None:
#             similarity_dot_12 = np.dot(emb1, emb2)
#             print(f"Similarity (1 vs 2) - Dot Product: {similarity_dot_12:.4f}")

#         # Batch embedding
#         batch_texts = [text1, text2, text3, "One more sentence.", invalid_text]
#         batch_embeddings = embedder.embed_batch(batch_texts)
#         print("Batch Embeddings Shape:", batch_embeddings.shape if batch_embeddings is not None else None)
