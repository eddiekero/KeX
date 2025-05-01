import sys
import subprocess
import json
import plyextract
import shutil
import visual_heatmap
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

#from skimage.metrics import structural_similarity as ssim


gaussian_counts = [100000, 250000, 500000, 750000]

init_train_iter = 100
iterations = [init_train_iter]
iteration_step = 500
for i in range(1, 20):
    iterations.append(iterations[-1] + iteration_step)
rendered_iterations = []
rendered_gaussian_counts = []
all_gaussian_counts = []

target = 0


# Set the time limit (15 minutes = 900 seconds)
TIME_LIMIT = 120 * 60
start_time = time.time()

# only use one scene for now
scene = "tandt_db/tandt/truck"

def InitTrainingRun(name):
    subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", f"output/{name}{iterations[0]}",
                        "-s", scene,
                        "--eval",
                        "--optimizer_type", "sparse_adam", 
                        "--iterations", f"{iterations[0]}", 
                        "--checkpoint_iterations", f"{iterations[0]}",])
    return

name = scene.split("/")[-1]

InitTrainingRun(name)
gaussian_count = plyextract.get_vertex_count(f"output/{name}{iterations[0]}/point_cloud/iteration_{init_train_iter}/point_cloud.ply")
all_gaussian_counts.append(gaussian_count)

checkpoint = 1
while (target < len(gaussian_counts)):
    # Check elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > TIME_LIMIT:
        print("Time limit exceeded. Stopping.")
        break

    subprocess.run([sys.executable,
                    "train.py", 
                    "--model_path", f"output/{name}{iterations[checkpoint]}",
                    "-s", scene,
                    "--eval",
                    "--optimizer_type", "sparse_adam",
                    "--start_checkpoint", f"output/{name}{iterations[checkpoint-1]}/chkpnt{iterations[checkpoint-1]}.pth",
                    "--iterations", f"{iterations[checkpoint]}", 
                    "--checkpoint_iterations", f"{iterations[checkpoint]}",])
    
    new_gaussian_count = plyextract.get_vertex_count(f"output/{name}{iterations[checkpoint]}/point_cloud/iteration_{iterations[checkpoint]}/point_cloud.ply")
    

    new_dist = np.abs(gaussian_counts[target] - new_gaussian_count)
    prev_dist = np.abs(gaussian_counts[target] - gaussian_count)
    
    if (new_dist > prev_dist): # The last iteration was closest to desired / target gaussian count
        subprocess.run([sys.executable,
                        "render.py", 
                        "-m", f"output/{name}{iterations[checkpoint-1]}",
                        "--skip_train",])
        rendered_iterations.append(iterations[checkpoint-1])
        rendered_gaussian_counts.append(gaussian_count)
        target+=1
    else: # remove 
        shutil.rmtree(f"output/{name}{iterations[checkpoint-1]}/")
    
    all_gaussian_counts.append(new_gaussian_count)
    gaussian_count = new_gaussian_count
    checkpoint+=1

save = {}
save['rendered_iterations'] = rendered_iterations
save['rendered_gaussian_counts'] = rendered_gaussian_counts
with open(f"output/{name}.pkl", "wb") as file:
    pickle.dump(save, file)







