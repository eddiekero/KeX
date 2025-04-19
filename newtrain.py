import sys
import subprocess
import json
import plyextract
import shutil

import pickle


iteration_step = 100
max_iteration = 4000
del_last_checkpoint = True
del_final_model = True

count_and_psnr = {}

scenes = [
    "tandt_db\db\playroom",
]

def InitTrainingRun():
    subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", "output/whatever0",
                        "-s", scene_dir,
                        "--eval",
                        "--optimizer_type", "sparse_adam",
                        "--iterations", f"{iteration_step}", 
                        "--checkpoint_iterations", f"{iteration_step}",])
    
    subprocess.run([sys.executable,
                        "render.py", 
                        "-m", "output/whatever0",
                        "--skip_train",])
        
    subprocess.run([sys.executable,
                    "metrics.py", 
                    "-m", "output/whatever0",])
    
    with open("output/whatever0/results.json", "r") as file:
        data = json.load(file)

    psnr = data[f"ours_{iteration_step}"]["PSNR"]

    gaussian_count = plyextract.get_vertex_count(f"output/whatever0/point_cloud/iteration_{iteration_step}/point_cloud.ply")

    count_and_psnr[scene_dir].append((gaussian_count, psnr))
    return


for scene_dir in scenes:

    print(f"\nOn scene {scene_dir}\n")
    
    count_and_psnr[scene_dir] = []
    
    
    iteration_step = 100
    InitTrainingRun()
    prev_iter = 100
    next_iter = 200

    checkpoint = 1
    
    while (next_iter < max_iteration):
        
        subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", "output/whatever%d" % (checkpoint),
                        "-s", scene_dir,
                        "--eval",
                        "--optimizer_type", "sparse_adam",
                        "--start_checkpoint", "output/whatever%d/chkpnt%d.pth" % (checkpoint-1, prev_iter),
                        "--iterations", "%d" % (next_iter), 
                        "--checkpoint_iterations", "%d" % (next_iter),])
        
        if (del_last_checkpoint):
            shutil.rmtree("output/whatever%d/" % (checkpoint-1))
        
        subprocess.run([sys.executable,
                        "render.py", 
                        "-m", "output/whatever%d" % checkpoint,
                        "--skip_train",])
        
        subprocess.run([sys.executable,
                        "metrics.py", 
                        "-m", "output/whatever%d" % checkpoint])
        
        with open("output/whatever%d/results.json" % checkpoint, "r") as file:
            data = json.load(file)

        psnr = data["ours_%d" % (next_iter)]["PSNR"]

        gaussian_count = plyextract.get_vertex_count("output/whatever%d/point_cloud/iteration_%d/point_cloud.ply" % (checkpoint, next_iter))

        count_and_psnr[scene_dir].append((gaussian_count, psnr))

        checkpoint+=1
        iteration_step *= 1.2
        iteration_step = int(iteration_step)
        prev_iter = next_iter
        next_iter += iteration_step

    print(f"{scene_dir}: count_and_psnr = {count_and_psnr[scene_dir]}")
    
    if (del_last_checkpoint):
        shutil.rmtree("output/whatever%d/" % (checkpoint-1))

with open("data.pkl", "wb") as file:
    pickle.dump(count_and_psnr, file)



