import os
import re

def get_vertex_count(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("element vertex"):
                return int(line.split()[2])  # Extract the number
    return None  # Return None if not found

def process_ply_files_recursively(folder_path):
    vertex_counts = {}
    for root, _, files in os.walk(folder_path):  # Walk through all subdirectories
        for file_name in files:
            if file_name.endswith(".ply"):
                file_path = os.path.join(root, file_name)
                subfolder_name = os.path.basename(root)  # Get the immediate parent folder
                iteration = re.findall('\d+', subfolder_name)[0]

                vertex_count = get_vertex_count(file_path)
                if vertex_count is not None:
                    vertex_counts[iteration] = vertex_counts.get(iteration, [])  # Initialize list if needed
                    vertex_counts[iteration].append((file_name, vertex_count))  # Store file name and vertex count
    return vertex_counts

# Example usage
folder_path = "output\\train4000\\point_cloud"  # Change this to your folder path
result = process_ply_files_recursively(folder_path)
print(result)
