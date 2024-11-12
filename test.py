import numpy as np

# Sample data for testing
datas = [
    {"key": np.array([[1, 2, 3], [4, 5, 6]])},  # shape (2, 3)
    {"key": np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])},  # shape (3, 3)
    {"key": np.array([[16, 17, 18]])},  # shape (1, 3)
]

data = {}

# Check whether all arrays under the key have the same size
for key in datas[0]:
    # Get the list of arrays for the current key from all episodes
    data_list = [b[key] for b in datas]
    
    # Check if all arrays in data_list have the same shape
    shapes = [arr.shape for arr in data_list]
    
    # If all shapes are the same, proceed with concatenation
    if all(shape == shapes[0] for shape in shapes):
        data[key] = np.concatenate(data_list, axis=0)
    else:
        print(f"Warning: Mismatched shapes for key '{key}': {shapes}")
        
        # Find the maximum number of rows
        max_rows = max([shape[0] for shape in shapes])
        
        # Pad each array to the maximum size by appending the last row
        for i, arr in enumerate(data_list):
            if arr.shape[0] < max_rows:
                # Get the last row of the array
                last_row = arr[-1]
                # Calculate how many rows to add
                rows_to_add = max_rows - arr.shape[0]
                # Create the padding by repeating the last row
                padding = np.tile(last_row, (rows_to_add, 1))
                # Append the padding to the original array
                data_list[i] = np.vstack([arr, padding])
        
        # After padding, concatenate the arrays
        data[key] = np.concatenate(data_list, axis=0)

# Display the result
print("Concatenated and padded result for 'key':")
print(data["key"])