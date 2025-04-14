import sys
import subprocess
import json
import plyextract
import shutil

import pickle


iteration_step = 100
max_iteration = 200
del_last_checkpoint = True
del_final_model = True

count_and_psnr = {}

scenes = [
    "tandt_db/tandt/truck/",
    "tandt_db/tandt/train",
]

def InitTrainingRun():
    subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", "output/whatever0",
                        "-s", scene_dir,
                        "--eval",
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

    InitTrainingRun()
    checkpoint = 1

    for iter in range(iteration_step, max_iteration, iteration_step):

        subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", "output/whatever%d" % (checkpoint),
                        "-s", scene_dir,
                        "--eval",
                        "--start_checkpoint", "output/whatever%d/chkpnt%d.pth" % (checkpoint-1, iter),
                        "--iterations", "%d" % (iter+iteration_step), 
                        "--checkpoint_iterations", "%d" % (iter+iteration_step),])
        
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

        psnr = data["ours_%d" % (iter+iteration_step)]["PSNR"]

        gaussian_count = plyextract.get_vertex_count("output/whatever%d/point_cloud/iteration_%d/point_cloud.ply" % (checkpoint, iter+iteration_step))

        count_and_psnr[scene_dir].append((gaussian_count, psnr))

        checkpoint+=1

    print(f"{scene_dir}: count_and_psnr = {count_and_psnr[scene_dir]}")
    
    if (del_last_checkpoint):
        shutil.rmtree("output/whatever%d/" % (checkpoint-1))



with open("data.pkl", "wb") as file:
    pickle.dump(count_and_psnr, file)



