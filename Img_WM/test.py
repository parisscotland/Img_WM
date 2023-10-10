def get_neighbors(arr, i, j, k):
    """
    Generate a list of neighboring locations around (i, j) within a range of k,
    avoiding out-of-bounds neighbors.

    Parameters:
    - arr: The 2-dimensional ndarray.
    - i: The row index.
    - j: The column index.
    - k: The neighborhood range.

    Returns:
    - A list of valid neighboring locations within the specified range.
    """
    neighbors = []
    rows, cols = arr.shape

    for row_offset in range(-k, k + 1):
        for col_offset in range(-k, k + 1):
            neighbor_i = i + row_offset
            neighbor_j = j + col_offset

            # Check if the neighbor is within bounds and not the same as the original location
            if (
                0 <= neighbor_i < rows
                and 0 <= neighbor_j < cols
                and (neighbor_i != i or neighbor_j != j)
            ):
                neighbors.append((neighbor_i, neighbor_j))

    return neighbors

# Example usage:
import numpy as np

# Create a 2D ndarray (replace this with your actual data)
ndarray_2d = np.array([[1, 2, 3, 1, 2, 3],
                       [4, 5, 6, 4, 5, 6],
                       [7, 8, 9, 7, 8, 9],
                       [1, 2, 3, 1, 2, 3],
                       [1, 2, 3, 1, 2, 3],])

# Specify the location (i, j) and neighborhood range (k)
i, j = 1, 1  # Example location
k = 3  # Example neighborhood range

# Get the list of neighboring locations
neighbors = get_neighbors(ndarray_2d, i, j, k)

# Print the list of neighboring locations
print(neighbors)
