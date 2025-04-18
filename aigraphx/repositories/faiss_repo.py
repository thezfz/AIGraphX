import logging
import os
import pickle
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
import faiss  # type: ignore[import-untyped]
import json
import asyncio
from typing import Literal
from unittest.mock import patch  # Keep patch if needed for error simulation

logger = logging.getLogger(__name__)


class FaissRepository:
    """Repository for interacting with a Faiss index."""

    def __init__(
        self,
        index_path: str = "data/papers_faiss.index",
        id_map_path: str = "data/papers_faiss_ids.json",
        id_type: Literal["int", "str"] = "int",
    ):
        """
        Initializes the FaissRepository by loading the index and ID map.

        Args:
            index_path: Path to the pre-built Faiss index file.
            id_map_path: Path to the JSON file mapping Faiss index positions to original IDs.
            id_type: The expected type of the original IDs ('int' or 'str').
        """
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[int, Union[int, str]] = {}
        self.id_type = id_type
        self._lock = asyncio.Lock()

        self._load_index()
        self._load_id_map()

    def _load_index(self) -> None:
        """Loads the Faiss index from the specified file."""
        if not os.path.exists(self.index_path):
            logger.error(
                f"Faiss index file not found at {self.index_path}. Search will not work."
            )
            self.index = None
            return

        try:
            logger.info(f"Loading Faiss index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            logger.info(
                f"Faiss index loaded successfully. Index contains {self.index.ntotal} vectors."
            )
            if self.index.ntotal == 0:
                logger.warning("Loaded Faiss index is empty.")
        except Exception as e:
            logger.error(
                f"Failed to load Faiss index from {self.index_path}: {e}", exc_info=True
            )
            self.index = None

    def _load_id_map(self) -> None:
        """Loads the ID map from the specified JSON file."""
        if not os.path.exists(self.id_map_path):
            logger.error(
                f"Faiss ID map file not found at {self.id_map_path}. Search result mapping will fail."
            )
            self.id_map = {}
            return

        try:
            logger.info(f"Loading Faiss ID map from {self.id_map_path}...")
            with open(self.id_map_path, "r") as f:
                # JSON keys are strings, convert them back to integers
                loaded_map_str_keys = json.load(f)
                # Convert values based on id_type
                value_converter = int if self.id_type == "int" else str
                self.id_map = {
                    int(k): value_converter(v) for k, v in loaded_map_str_keys.items()
                }
            logger.info(
                f"Faiss ID map loaded successfully. Map contains {len(self.id_map)} entries."
            )
        except (json.JSONDecodeError, ValueError, IOError) as e:
            logger.error(
                f"Failed to load or parse Faiss ID map from {self.id_map_path}: {e}",
                exc_info=True,
            )
            self.id_map = {}

    def is_ready(self) -> bool:
        """Check if the index and ID map are loaded successfully."""
        # Add detailed logging
        index_exists = self.index is not None
        ntotal = (
            self.index.ntotal if self.index else -1
        )  # Use -1 to indicate index is None
        map_exists_and_not_empty = bool(self.id_map)
        ready_status = index_exists and ntotal > 0 and map_exists_and_not_empty
        logger.info(
            f"[is_ready Check - Instance ID: {id(self)}] index_exists={index_exists}, index.ntotal={ntotal}, map_exists_and_not_empty={map_exists_and_not_empty} -> Returning: {ready_status}"
        )
        # Original logic:
        # return self.index is not None and self.index.ntotal > 0 and bool(self.id_map)
        return ready_status

    async def search_similar(
        self, embedding: np.ndarray, k: int = 10
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Searches the Faiss index for vectors similar to the query embedding.

        Args:
            embedding: The query embedding vector (numpy array, float32).
            k: The number of nearest neighbors to retrieve.

        Returns:
            A list of tuples, where each tuple contains (original_id, distance).
            original_id type depends on id_type specified during initialization.
            Returns an empty list if the index is not ready or search fails.
        """
        if not self.is_ready():
            logger.warning(
                "Faiss index or ID map not ready. Returning empty search results."
            )
            return []

        if self.index is None:
            logger.error("Search attempt while index is None.")
            return []

        if not isinstance(embedding, np.ndarray):
            logger.error("Invalid query embedding type. Expected numpy array.")
            return []
        if embedding.ndim == 1:
            query_vector = embedding.reshape(1, -1).astype(np.float32)
        elif embedding.ndim == 2 and embedding.shape[0] == 1:
            query_vector = embedding.astype(np.float32)
        else:
            logger.error(
                f"Invalid query embedding shape: {embedding.shape}. Expected (dim,) or (1, dim)."
            )
            return []

        if query_vector.shape[1] != self.index.d:
            logger.error(
                f"Query embedding dimension ({query_vector.shape[1]}) does not match index dimension ({self.index.d})."
            )
            return []

        actual_k = min(k, self.index.ntotal)
        if actual_k <= 0:
            logger.warning("Search k is 0 or index is empty. Returning empty results.")
            return []

        logger.debug(f"Performing Faiss search for {actual_k} neighbors...")
        try:
            async with self._lock:
                distances, indices = await asyncio.to_thread(
                    self.index.search, query_vector, actual_k
                )

            results: List[Tuple[Union[int, str], float]] = []
            if indices.size > 0 and distances.size > 0:
                faiss_indices = indices[0]
                faiss_distances = distances[0]

                for i, faiss_idx in enumerate(faiss_indices):
                    if faiss_idx == -1:
                        logger.warning(
                            f"Faiss search returned invalid index -1 at position {i}. Skipping."
                        )
                        continue
                    # Get original_id from map (type is Union[int, str])
                    original_id = self.id_map.get(int(faiss_idx))
                    if original_id is not None:
                        distance = float(faiss_distances[i])
                        results.append((original_id, distance))
                    else:
                        logger.warning(
                            f"Could not find original_id in id_map for Faiss index {faiss_idx}. Skipping."
                        )

            logger.debug(f"Faiss search completed. Found {len(results)} valid results.")
            return results

        except Exception as e:
            logger.error(f"Error during Faiss search: {e}", exc_info=True)
            return []

    def get_index_size(self) -> int:
        """Returns the number of vectors currently in the index."""
        return self.index.ntotal if self.index else 0
