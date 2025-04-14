import json
import matplotlib.pyplot as plt
import plyextract

import pickle

# Plot the data
plt.figure(figsize=(8, 5))

# Read JSON data from a file

# To read it back
with open("data.pkl", "rb") as file:
    loaded_dict = pickle.load(file)


for key in loaded_dict.keys():
    # Extract X and Y values
    x_values, y_values = zip(*loaded_dict[key])

    # Sort values for correct plotting order
    sorted_indices = sorted(range(len(x_values)), key=lambda i: x_values[i])
    x_values = [x_values[i] for i in sorted_indices]
    y_values = [y_values[i] for i in sorted_indices]

    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Train')

#############################################
# with open("output/truck4000/results.json", "r") as file:
#     data = json.load(file)

# # Extract X and Y values
# x_values = [int(key.split('_')[1]) for key in data.keys()]
# y_values = [data[key]["PSNR"] for key in data.keys()]

# # Sort values for correct plotting order
# sorted_indices = sorted(range(len(x_values)), key=lambda i: x_values[i])
# x_values = [x_values[i] for i in sorted_indices]
# y_values = [y_values[i] for i in sorted_indices]


# folder_path = "output\\truck4000\\point_cloud"  # Change this to your folder path
# vertex_counts = plyextract.process_ply_files_recursively(folder_path)
# for idx, x in enumerate(x_values):
#     x_values[idx] = vertex_counts[str(x)][0][1]



# plt.plot(x_values, y_values, marker='o', linestyle='-', color='r', label='Truck')
plt.xlabel("#Gaussians")
plt.ylabel("PSNR")
plt.title("combined truck and train")
plt.legend()
plt.grid(True)
plt.savefig("plot")
plt.show()