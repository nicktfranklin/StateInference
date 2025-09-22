"""
Finite Scalar Quantization (FSQ) implementation.

Based on "Finite Scalar Quantization: VQ-VAE Made Simple"
https://arxiv.org/abs/2309.15505
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


class FSQ(nn.Module):
    """
    Finite Scalar Quantization layer.
    
    Args:
        levels: List of integers specifying the number of levels for each dimension.
                For example, [8, 5, 5, 5] creates a codebook with 8*5*5*5 = 1000 codes.
    """
    
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        
        # Register levels as buffer so it moves with the model
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        
        # Calculate total number of codes
        self.codebook_size = 1
        for level in levels:
            self.codebook_size *= level
            
        # Pre-compute the bound for each dimension
        # Each dimension is quantized to [-bound, -bound+2, ..., bound-2, bound] 
        # with `level` total values
        bounds = []
        for level in levels:
            if level == 1:
                bounds.append(0.0)
            else:
                bound = level // 2
                bounds.append(float(bound))
        
        self.register_buffer('_bounds', torch.tensor(bounds, dtype=torch.float32))
        
    def bound(self) -> Tensor:
        """Get the bounds for each dimension."""
        return self._bounds
    
    def levels_tensor(self) -> Tensor:
        """Get the levels as a tensor."""
        return self._levels
    
    def quantize(self, z: Tensor) -> Tensor:
        """
        Quantize continuous values to discrete levels.
        
        Args:
            z: Input tensor of shape (..., dim)
            
        Returns:
            Quantized tensor of same shape as input
        """
        # Ensure z has the right number of dimensions
        assert z.shape[-1] == self.dim, f"Expected {self.dim} dimensions, got {z.shape[-1]}"
        
        quantized = torch.zeros_like(z)
        
        for i, (level, bound) in enumerate(zip(self.levels, self.bound())):
            if level == 1:
                quantized[..., i] = 0
            else:
                # Quantize to {-bound, -bound+2, ..., bound-2, bound}
                # First clamp to [-bound, bound]
                clamped = torch.clamp(z[..., i], -bound, bound)
                # Round to nearest even integer (then shift to get the right levels)
                quantized[..., i] = torch.round(clamped)
                # Ensure we stay within bounds after rounding
                quantized[..., i] = torch.clamp(quantized[..., i], -bound, bound)
        
        return quantized
    
    def dequantize(self, indices: Tensor) -> Tensor:
        """
        Convert indices back to quantized values.
        
        Args:
            indices: Integer indices of shape (...,)
            
        Returns:
            Quantized values of shape (..., dim)
        """
        # Convert flat indices to multi-dimensional indices
        batch_shape = indices.shape
        indices_flat = indices.view(-1)
        
        # Compute multi-dimensional indices
        multi_indices = []
        remaining = indices_flat
        
        for level in reversed(self.levels):
            multi_indices.append(remaining % level)
            remaining = remaining // level
        
        multi_indices.reverse()
        multi_indices = torch.stack(multi_indices, dim=-1)  # Shape: (batch_size_flat, dim)
        
        # Convert to quantized values
        quantized = torch.zeros_like(multi_indices, dtype=torch.float32)
        
        for i, (level, bound) in enumerate(zip(self.levels, self.bound())):
            if level == 1:
                quantized[..., i] = 0
            else:
                # Map indices [0, 1, ..., level-1] to [-bound, -bound+2, ..., bound-2, bound]
                quantized[..., i] = multi_indices[..., i].float() * 2 - bound
        
        # Reshape back to original batch shape + dim
        return quantized.view(*batch_shape, self.dim)
    
    def indices_to_codes(self, z_quantized: Tensor) -> Tensor:
        """
        Convert quantized values to flat indices.
        
        Args:
            z_quantized: Quantized values of shape (..., dim)
            
        Returns:
            Flat indices of shape (...,)
        """
        assert z_quantized.shape[-1] == self.dim
        
        batch_shape = z_quantized.shape[:-1]
        z_flat = z_quantized.view(-1, self.dim)
        
        # Convert quantized values to indices
        indices = torch.zeros(z_flat.shape[0], dtype=torch.long, device=z_quantized.device)
        
        for i, (level, bound) in enumerate(zip(self.levels, self.bound())):
            if level == 1:
                dim_indices = torch.zeros_like(z_flat[:, i], dtype=torch.long)
            else:
                # Map [-bound, -bound+2, ..., bound-2, bound] to [0, 1, ..., level-1]
                dim_indices = ((z_flat[:, i] + bound) / 2).round().long()
                dim_indices = torch.clamp(dim_indices, 0, level - 1)
            
            # Add contribution of this dimension to flat index
            multiplier = 1
            for j in range(i + 1, len(self.levels)):
                multiplier *= self.levels[j]
            
            indices += dim_indices * multiplier
        
        return indices.view(*batch_shape)
    
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with straight-through estimator.
        
        Args:
            z: Input tensor of shape (..., dim)
            
        Returns:
            Tuple of (quantized_values, indices)
        """
        # Quantize
        z_quantized = self.quantize(z)
        
        # Straight-through estimator: use quantized values in forward pass
        # but use original gradients in backward pass
        z_quantized = z + (z_quantized - z).detach()
        
        # Get indices for codebook lookup if needed
        indices = self.indices_to_codes(z_quantized)
        
        return z_quantized, indices


def calculate_codebook_size(levels: List[int]) -> int:
    """Calculate the total codebook size for given levels."""
    size = 1
    for level in levels:
        size *= level
    return size


def suggest_fsq_levels(target_codebook_size: int, dim: int) -> List[int]:
    """
    Suggest FSQ levels that approximate a target codebook size.
    
    Args:
        target_codebook_size: Desired number of codes
        dim: Number of dimensions
        
    Returns:
        List of levels that gives a codebook size close to the target
    """
    # Start with uniform levels
    base_level = int(target_codebook_size ** (1.0 / dim))
    levels = [base_level] * dim
    
    # Adjust to get closer to target
    current_size = calculate_codebook_size(levels)
    
    # If we're under the target, try increasing some dimensions
    if current_size < target_codebook_size:
        for i in range(dim):
            test_levels = levels.copy()
            test_levels[i] += 1
            test_size = calculate_codebook_size(test_levels)
            if test_size <= target_codebook_size * 1.2:  # Allow 20% overshoot
                levels[i] += 1
                current_size = test_size
                if current_size >= target_codebook_size:
                    break
    
    return levels