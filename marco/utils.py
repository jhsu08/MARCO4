"""
Utility Functions for MARCO4

Contains helper functions for grid manipulation, serialization,
and other common operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
import hashlib
import json
from copy import deepcopy


# Grid type alias
Grid = np.ndarray


def create_empty_grid(height: int, width: int) -> Grid:
    """
    Create an empty grid (all cells marked as -1).

    Args:
        height: Number of rows
        width: Number of columns

    Returns:
        numpy array of shape (height, width) filled with -1
    """
    return np.full((height, width), -1, dtype=np.int8)


def is_complete_grid(grid: Grid) -> bool:
    """
    Check if a grid is complete (no -1 cells).

    Args:
        grid: The grid to check

    Returns:
        True if all cells are filled (no -1 values)
    """
    return not np.any(grid == -1)


def is_partial_grid(grid: Grid) -> bool:
    """
    Check if a grid is partial (has at least one -1 cell).

    Args:
        grid: The grid to check

    Returns:
        True if at least one cell is -1
    """
    return np.any(grid == -1)


def count_empty_cells(grid: Grid) -> int:
    """
    Count the number of empty (-1) cells in a grid.

    Args:
        grid: The grid to analyze

    Returns:
        Number of cells with value -1
    """
    return int(np.sum(grid == -1))


def count_filled_cells(grid: Grid) -> int:
    """
    Count the number of filled (non -1) cells in a grid.

    Args:
        grid: The grid to analyze

    Returns:
        Number of cells with value >= 0
    """
    return int(np.sum(grid >= 0))


def get_empty_cell_positions(grid: Grid) -> List[Tuple[int, int]]:
    """
    Get positions of all empty cells in row-major order.

    Args:
        grid: The grid to analyze

    Returns:
        List of (row, col) tuples for empty cells
    """
    positions = np.argwhere(grid == -1)
    return [(int(pos[0]), int(pos[1])) for pos in positions]


def get_filled_cell_positions(grid: Grid) -> List[Tuple[int, int]]:
    """
    Get positions of all filled cells.

    Args:
        grid: The grid to analyze

    Returns:
        List of (row, col) tuples for filled cells
    """
    positions = np.argwhere(grid >= 0)
    return [(int(pos[0]), int(pos[1])) for pos in positions]


def get_next_empty_cell(grid: Grid) -> Optional[Tuple[int, int]]:
    """
    Get the next empty cell in row-major order.

    Args:
        grid: The grid to analyze

    Returns:
        (row, col) tuple of first empty cell, or None if grid is complete
    """
    positions = np.argwhere(grid == -1)
    if len(positions) == 0:
        return None
    return (int(positions[0][0]), int(positions[0][1]))


def grid_to_hash(grid: Grid) -> str:
    """
    Create a hash of a grid for deduplication.

    Args:
        grid: The grid to hash

    Returns:
        SHA-256 hash string
    """
    return hashlib.sha256(grid.tobytes()).hexdigest()[:16]


def grid_to_string(grid: Grid, show_empty: bool = True) -> str:
    """
    Convert grid to human-readable string.

    Args:
        grid: The grid to convert
        show_empty: If True, show -1 as '.', otherwise as '-1'

    Returns:
        Multi-line string representation
    """
    lines = []
    for row in grid:
        row_str = []
        for val in row:
            if val == -1:
                row_str.append('.' if show_empty else '-1')
            else:
                row_str.append(str(val))
        lines.append(' '.join(row_str))
    return '\n'.join(lines)


def string_to_grid(s: str) -> Grid:
    """
    Parse grid from string representation.

    Args:
        s: Multi-line string with space-separated values

    Returns:
        numpy array grid
    """
    lines = s.strip().split('\n')
    rows = []
    for line in lines:
        row = []
        for val in line.split():
            if val == '.':
                row.append(-1)
            else:
                row.append(int(val))
        rows.append(row)
    return np.array(rows, dtype=np.int8)


def grid_to_json(grid: Grid) -> str:
    """
    Serialize grid to JSON string.

    Args:
        grid: The grid to serialize

    Returns:
        JSON string
    """
    return json.dumps(grid.tolist())


def json_to_grid(s: str) -> Grid:
    """
    Deserialize grid from JSON string.

    Args:
        s: JSON string

    Returns:
        numpy array grid
    """
    return np.array(json.loads(s), dtype=np.int8)


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """
    Check if two grids are equal.

    Args:
        g1, g2: Grids to compare

    Returns:
        True if grids have same shape and values
    """
    if g1.shape != g2.shape:
        return False
    return np.array_equal(g1, g2)


def copy_grid(grid: Grid) -> Grid:
    """
    Create a deep copy of a grid.

    Args:
        grid: The grid to copy

    Returns:
        New numpy array with same values
    """
    return grid.copy()


def set_cell(grid: Grid, row: int, col: int, value: int) -> Grid:
    """
    Create a new grid with one cell changed.

    Args:
        grid: The original grid
        row, col: Position to change
        value: New value (0-9 or -1)

    Returns:
        New grid with the cell set
    """
    new_grid = grid.copy()
    new_grid[row, col] = value
    return new_grid


def merge_grids(base: Grid, overlay: Grid) -> Grid:
    """
    Merge two grids, taking non-empty values from overlay.

    Args:
        base: Base grid
        overlay: Grid with values to overlay

    Returns:
        New grid with overlay values where overlay is non-empty
    """
    if base.shape != overlay.shape:
        raise ValueError(f"Grid shapes don't match: {base.shape} vs {overlay.shape}")

    result = base.copy()
    mask = overlay >= 0
    result[mask] = overlay[mask]
    return result


def get_grid_colors(grid: Grid) -> Set[int]:
    """
    Get unique colors used in a grid (excluding -1).

    Args:
        grid: The grid to analyze

    Returns:
        Set of color values
    """
    unique = np.unique(grid)
    return set(int(v) for v in unique if v >= 0)


def validate_grid(grid: Grid, allow_empty: bool = True) -> bool:
    """
    Validate that a grid has valid color values.

    Args:
        grid: The grid to validate
        allow_empty: If True, allow -1 values

    Returns:
        True if valid
    """
    if grid.ndim != 2:
        return False

    if allow_empty:
        # All values should be -1 or 0-9
        return np.all((grid >= -1) & (grid <= 9))
    else:
        # All values should be 0-9
        return np.all((grid >= 0) & (grid <= 9))


# Grid augmentation functions
def rotate_90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def flip_horizontal(grid: Grid) -> Grid:
    """Flip grid horizontally (left-right)."""
    return np.fliplr(grid)


def flip_vertical(grid: Grid) -> Grid:
    """Flip grid vertically (up-down)."""
    return np.flipud(grid)


def flip_diagonal(grid: Grid) -> Grid:
    """Flip grid along main diagonal (transpose)."""
    return grid.T


def flip_antidiagonal(grid: Grid) -> Grid:
    """Flip grid along anti-diagonal."""
    return np.rot90(grid.T, k=2)


def apply_augmentation(grid: Grid, aug_name: str) -> Grid:
    """
    Apply a named augmentation to a grid.

    Args:
        grid: The grid to augment
        aug_name: Name of augmentation

    Returns:
        Augmented grid
    """
    augmentations = {
        'identity': lambda g: g.copy(),
        'rot90': rotate_90,
        'rot180': rotate_180,
        'rot270': rotate_270,
        'flip_h': flip_horizontal,
        'flip_v': flip_vertical,
        'flip_diag': flip_diagonal,
        'flip_antidiag': flip_antidiagonal,
        'rot90_flip_h': lambda g: flip_horizontal(rotate_90(g)),
        'rot90_flip_v': lambda g: flip_vertical(rotate_90(g)),
        'rot180_flip_h': lambda g: flip_horizontal(rotate_180(g)),
        'rot180_flip_v': lambda g: flip_vertical(rotate_180(g)),
        'rot270_flip_h': lambda g: flip_horizontal(rotate_270(g)),
        'rot270_flip_v': lambda g: flip_vertical(rotate_270(g)),
        'transpose': flip_diagonal,
        'transpose_rot180': lambda g: rotate_180(flip_diagonal(g)),
    }

    if aug_name not in augmentations:
        raise ValueError(f"Unknown augmentation: {aug_name}")

    return augmentations[aug_name](grid)


def get_inverse_augmentation(aug_name: str) -> str:
    """
    Get the inverse of an augmentation.

    Args:
        aug_name: Name of augmentation

    Returns:
        Name of inverse augmentation
    """
    inverses = {
        'identity': 'identity',
        'rot90': 'rot270',
        'rot180': 'rot180',
        'rot270': 'rot90',
        'flip_h': 'flip_h',
        'flip_v': 'flip_v',
        'flip_diag': 'flip_diag',
        'flip_antidiag': 'flip_antidiag',
        'rot90_flip_h': 'flip_h',  # flip_h then rot270
        'rot90_flip_v': 'flip_v',
        'rot180_flip_h': 'flip_h',
        'rot180_flip_v': 'flip_v',
        'rot270_flip_h': 'flip_h',
        'rot270_flip_v': 'flip_v',
        'transpose': 'transpose',
        'transpose_rot180': 'transpose_rot180',
    }

    if aug_name not in inverses:
        raise ValueError(f"Unknown augmentation: {aug_name}")

    return inverses[aug_name]


def compute_grid_similarity(g1: Grid, g2: Grid) -> float:
    """
    Compute similarity between two grids.

    Args:
        g1, g2: Grids to compare (must have same shape)

    Returns:
        Similarity score in [0, 1]
    """
    if g1.shape != g2.shape:
        return 0.0

    total_cells = g1.size
    if total_cells == 0:
        return 1.0

    # Count matching cells (excluding -1 comparisons)
    matching = np.sum((g1 == g2) & (g1 >= 0) & (g2 >= 0))
    comparable = np.sum((g1 >= 0) & (g2 >= 0))

    if comparable == 0:
        return 0.5  # No cells to compare

    return matching / comparable


def extract_subgrid(grid: Grid, row_start: int, row_end: int,
                    col_start: int, col_end: int) -> Grid:
    """
    Extract a subgrid from a grid.

    Args:
        grid: Source grid
        row_start, row_end: Row range (exclusive end)
        col_start, col_end: Column range (exclusive end)

    Returns:
        Subgrid as new numpy array
    """
    return grid[row_start:row_end, col_start:col_end].copy()


def pad_grid(grid: Grid, target_height: int, target_width: int,
             fill_value: int = -1) -> Grid:
    """
    Pad a grid to target dimensions.

    Args:
        grid: Source grid
        target_height: Target height
        target_width: Target width
        fill_value: Value to fill padding with (default -1)

    Returns:
        Padded grid
    """
    h, w = grid.shape
    if h >= target_height and w >= target_width:
        return grid.copy()

    result = np.full((max(h, target_height), max(w, target_width)),
                     fill_value, dtype=np.int8)
    result[:h, :w] = grid
    return result


class GridEncoder:
    """
    Encode/decode grids for neural network input.
    """

    def __init__(self, max_height: int = 30, max_width: int = 30):
        self.max_height = max_height
        self.max_width = max_width
        self.num_colors = 11  # 0-9 plus empty (-1 -> 10)

    def encode(self, grid: Grid) -> np.ndarray:
        """
        Encode grid as one-hot tensor.

        Args:
            grid: Grid to encode

        Returns:
            Tensor of shape (num_colors, max_height, max_width)
        """
        h, w = grid.shape
        encoded = np.zeros((self.num_colors, self.max_height, self.max_width),
                           dtype=np.float32)

        for i in range(min(h, self.max_height)):
            for j in range(min(w, self.max_width)):
                val = grid[i, j]
                if val == -1:
                    encoded[10, i, j] = 1.0
                else:
                    encoded[val, i, j] = 1.0

        return encoded

    def decode(self, encoded: np.ndarray, height: int, width: int) -> Grid:
        """
        Decode one-hot tensor back to grid.

        Args:
            encoded: Tensor of shape (num_colors, height, width)
            height: Target height
            width: Target width

        Returns:
            Decoded grid
        """
        grid = np.full((height, width), -1, dtype=np.int8)

        for i in range(height):
            for j in range(width):
                color = np.argmax(encoded[:, i, j])
                if color == 10:
                    grid[i, j] = -1
                else:
                    grid[i, j] = color

        return grid
