import torch
from torch import Tensor


class TensorIndexer:
    def __init__(self, tensors=None, device=None):
        """
        Initialize the indexer with optional tensors and device.
        Args:
            tensors: Optional list of tensors, each of shape (d) containing flattened one-hot vectors
            device: Optional torch.device. If None, uses CUDA if available, else CPU
        """
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.reset()

        if tensors is not None:
            self.add(tensors)

    @property
    def n(self):
        """Number of unique vectors stored in the indexer"""
        return self.reference_tensors.shape[0]

    def reset(self):
        """Reset the indexer to an empty state"""
        self.reference_tensors = torch.empty(
            (0, 0), dtype=torch.long, device=self.device
        )

    @torch.no_grad()
    def contains(self, query_tensor):
        """
        Check if vector(s) exist in the indexer.
        Args:
            query_tensor: Tensor of shape (d) or (n,d) of type torch.long
        Returns:
            bool or torch.Tensor: Boolean for single vector, boolean tensor of shape (n,) for batch
        """
        if self.n == 0:
            return (
                False
                if query_tensor.dim() == 1
                else torch.zeros(
                    len(query_tensor), dtype=torch.bool, device=self.device
                )
            )

        # Ensure query is 2D
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)

        # Check for matches using all dimensions
        matches = torch.all(
            self.reference_tensors.unsqueeze(1) == query_tensor.unsqueeze(0), dim=-1
        )  # shape: (num_stored, num_queries)
        return torch.any(matches, dim=0)

    @torch.no_grad()
    def get_state_hash(self, tensor) -> Tensor:
        """Returns the index of the tensor in the reference_tensors, -1 if not found"""

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.long)

        # handle empty indexer case
        if self.n == 0:
            return torch.full((len(tensor),), -1, device=self.device, dtype=torch.long)

        # Find which vectors already exist
        exists = self.contains(tensor)

        # Get only the new unique vectors
        indicies = torch.full((len(tensor),), -1, device=self.device)
        for i in range(len(tensor)):
            matches = torch.all(self.reference_tensors == tensor[i], dim=1)
            if torch.any(matches):
                indicies[i] = torch.where(matches)[0][0]

        return indicies[0] if len(tensor) == 1 else indicies

    @torch.no_grad()
    def add(self, tensor):
        """
        Add new unique vector(s) to the indexer.
        Args:
            tensor: Tensor of shape (d) or (n,d) of type torch.long
        Returns:
            int or torch.Tensor: Index/indices of the vectors (-1 for duplicates)
        """
        # Convert to 2D if necessary
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        # Move to correct device
        tensor = tensor.to(device=self.device, dtype=torch.long)

        # Handle empty indexer case
        if self.n == 0:
            # Find unique vectors in the input
            unique_vectors = torch.unique(tensor, dim=0)
            self.reference_tensors = unique_vectors

            # Map input vectors to their indices
            indices = torch.full((len(tensor),), -1, device=self.device)
            for i in range(len(tensor)):
                matches = torch.all(unique_vectors == tensor[i], dim=1)
                indices[i] = torch.where(matches)[0][0]
            return indices[0] if len(tensor) == 1 else indices

        # Find which vectors already exist
        exists = self.contains(tensor)

        # Get only the new unique vectors
        new_vectors = tensor[~exists]
        if len(new_vectors) > 0:
            unique_new = torch.unique(new_vectors, dim=0)
            self.reference_tensors = torch.cat(
                [self.reference_tensors, unique_new], dim=0
            )

        # Build indices for all input vectors
        indices = torch.full((len(tensor),), -1, device=self.device)
        for i in range(len(tensor)):
            matches = torch.all(self.reference_tensors == tensor[i], dim=1)
            indices[i] = torch.where(matches)[0][0]

        return indices[0] if len(tensor) == 1 else indices

    @torch.no_grad()
    def __call__(self, query_tensor):
        """
        Get indices for vector(s), adding new unique ones if they don't exist.
        Args:
            query_tensor: Tensor of shape (d) or (n,d) of type torch.long
        Returns:
            torch.Tensor: Scalar tensor for (d) input or tensor of shape (n,) for (n,d) input
        """
        return self.get_state_hash(query_tensor)

    def lookup(self, indices):
        """
        Look up original vectors from their indices.
        Args:
            indices: Integer or tensor of indices
        Returns:
            torch.Tensor: Original vector(s) of shape (d) for single index or (n,d) for multiple indices
        Raises:
            IndexError: If any index is out of range
        """
        if self.n == 0:
            raise IndexError("Indexer is empty")

        if isinstance(indices, int):
            if not 0 <= indices < self.n:
                raise IndexError(f"Index {indices} is out of range [0, {self.n})")
            return self.reference_tensors[indices]

        indices = torch.as_tensor(indices, device=self.device)
        if not torch.all((0 <= indices) & (indices < self.n)):
            raise IndexError(f"Some indices are out of range [0, {self.n})")

        return self.reference_tensors[indices]


# Example usage:
def test_indexer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create flattened one-hot vectors
    d = 6  # b*n = 2*3 = 6
    t1 = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device)
    t2 = torch.tensor([0, 1, 0, 1, 0, 0], dtype=torch.float32, device=device)
    t3 = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.float32, device=device)

    indexer = TensorIndexer(device=device)

    # Test single vectors
    idx1 = indexer(t1)
    idx2 = indexer(t2)
    idx3 = indexer(t3)
    print("Indices:", idx1, idx2, idx3)

    # Test batch
    batch = torch.stack([t1, t2, t3])
    batch_indices = indexer(batch)
    print("Batch indices:", batch_indices)

    # Test lookup
    recovered = indexer.lookup(batch_indices)
    print("All vectors match:", torch.all(batch == recovered))

    # Test new vector
    t4 = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32, device=device)
    idx4 = indexer(t4)
    print("New vector index:", idx4)


if __name__ == "__main__":
    test_indexer()
