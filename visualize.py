import json
import matplotlib.pyplot as plt
import plyextract


# Read JSON data from a file
with open("output/train4000/results.json", "r") as file:
    data = json.load(file)

# Extract X and Y values
x_values = [int(key.split('_')[1]) for key in data.keys()]
y_values = [data[key]["PSNR"] for key in data.keys()]

# Sort values for correct plotting order
sorted_indices = sorted(range(len(x_values)), key=lambda i: x_values[i])
x_values = [x_values[i] for i in sorted_indices]
y_values = [y_values[i] for i in sorted_indices]


folder_path = "output\\train4000\\point_cloud"  # Change this to your folder path
vertex_counts = plyextract.process_ply_files_recursively(folder_path)
for idx, x in enumerate(x_values):
    x_values[idx] = vertex_counts[str(x)][0][1]


# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='PSNR')
plt.xlabel("#Gaussians")
plt.ylabel("PSNR")
plt.title("Train4000")
plt.legend()
plt.grid(True)
plt.show()