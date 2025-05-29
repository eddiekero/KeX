import sys
import subprocess
import json
import plyextract
import shutil
import os
import pickle


count_and_psnr = {}

scenes = [
    "Mipnerf/bicycle",
    "Mipnerf/bonsai",
    "Mipnerf/counter",
    "Mipnerf/garden",
]

iterations = [100]
iteration_step = 500
for i in range(1, 50):
    iterations.append(iterations[-1] + iteration_step)
    iteration_step*=1.2
    iteration_step = int(iteration_step)

max_gaussian_count = 750000
all_gaussian_counts = []

def InitTrainingRun(scene, save_name):
    subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", f"output/{save_name}{iterations[0]}",
                        "-s", scene,
                        "--eval",
                        "--optimizer_type", "sparse_adam", 
                        "--iterations", f"{iterations[0]}", 
                        "--checkpoint_iterations", f"{iterations[0]}",])
    return


for scene_dir in scenes:
    print(f"\nOn scene {scene_dir}\n")
    
    save_name = scene_dir.split("/")[-1]
    count_and_psnr[save_name] = []

    if not os.path.isdir(f"output/{save_name}{iterations[0]}"):
        InitTrainingRun(scene_dir, save_name)
    else:
        print(f"Skipping training {save_name}{iterations[0]} since it already exists")

    gaussian_count = plyextract.get_vertex_count(f"output/{save_name}{iterations[0]}/point_cloud/iteration_{iterations[0]}/point_cloud.ply")
    all_gaussian_counts.append(gaussian_count)
    

    checkpoint = 1
    while (gaussian_count < max_gaussian_count):
        print(f"on checkpoint={checkpoint}")

        if not os.path.isdir(f"output/{save_name}{iterations[checkpoint]}"):
            subprocess.run([sys.executable,
                            "train.py", 
                            "--model_path", f"output/{save_name}{iterations[checkpoint]}",
                            "-s", scene_dir,
                            "--eval",
                            "--optimizer_type", "sparse_adam",
                            "--start_checkpoint", f"output/{save_name}{iterations[checkpoint-1]}/chkpnt{iterations[checkpoint-1]}.pth",
                            "--iterations", f"{iterations[checkpoint]}", 
                            "--checkpoint_iterations", f"{iterations[checkpoint]}"])
        else:
            print(f"Skipping training {save_name}{iterations[checkpoint]} since it already exists")
        
        # #shutil.rmtree(f"output/{save_name}{iterations[checkpoint-1]}/")
        
        # subprocess.run([sys.executable,
        #                 "render.py", 
        #                 "-m", f"output/{save_name}{iterations[checkpoint]}",
        #                 "--skip_train",])
        
        # subprocess.run([sys.executable,
        #                 "metrics.py", 
        #                 "-m", f"output/{save_name}{iterations[checkpoint]}"])
        
        # with open(f"output/{save_name}{iterations[checkpoint]}/results.json", "r") as file:
        #     data = json.load(file)

        # psnr = data[f"ours_{iterations[checkpoint]}"]["PSNR"]

        gaussian_count = plyextract.get_vertex_count(f"output/{save_name}{iterations[checkpoint]}/point_cloud/iteration_{iterations[checkpoint]}/point_cloud.ply")
        all_gaussian_counts.append(gaussian_count)
        
        #print((gaussian_count, psnr))
        print(gaussian_count)

        #count_and_psnr[save_name].append((gaussian_count, psnr))

        checkpoint+=1

    #print(f"{scene_dir}: count_and_psnr = {count_and_psnr[scene_dir]}")
    print(all_gaussian_counts)
    print(iterations[:checkpoint])
    #shutil.rmtree(f"output/{save_name}{iterations[checkpoint-1]}/")

with open("data.pkl", "wb") as file:
    pickle.dump(count_and_psnr, file)



