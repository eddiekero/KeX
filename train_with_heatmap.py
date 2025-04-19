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

iterations = [100, 2500, 5000]
doTrain = True
del_last_checkpoint = False
del_final_model = False

count_and_psnr = {}

scenes = [
    "tandt_db/tandt/train",
]

def InitTrainingRun():
    subprocess.run([sys.executable,
                        "train.py", 
                        "--model_path", "output/whatever0",
                        "-s", scene_dir,
                        "--eval",
                        "--optimizer_type", "sparse_adam", 
                        "--iterations", f"{iterations[0]}", 
                        "--checkpoint_iterations", f"{iterations[0]}",])
    
    subprocess.run([sys.executable,
                        "render.py", 
                        "-m", "output/whatever0",
                        "--skip_train",])
        
    # subprocess.run([sys.executable,
    #                 "metrics.py", 
    #                 "-m", "output/whatever0",])
    
    # with open("output/whatever0/results.json", "r") as file:
    #     data = json.load(file)

    # psnr = data[f"ours_{iterations[0]}"]["PSNR"]

    # gaussian_count = plyextract.get_vertex_count(f"output/whatever0/point_cloud/iteration_{iterations[0]}/point_cloud.ply")

    # count_and_psnr[scene_dir].append((gaussian_count, psnr))
    return

if (doTrain):
    for scene_dir in scenes:

        print(f"\nOn scene {scene_dir}\n")
        
        count_and_psnr[scene_dir] = []
        
        InitTrainingRun()
        

        checkpoint = 1
        
        for i, iter in enumerate(iterations):
            if (i == 0):
                continue
            
            subprocess.run([sys.executable,
                            "train.py", 
                            "--model_path", "output/whatever%d" % (checkpoint),
                            "-s", scene_dir,
                            "--eval",
                            "--optimizer_type", "sparse_adam",
                            "--start_checkpoint", "output/whatever%d/chkpnt%d.pth" % (checkpoint-1, iterations[i-1]),
                            "--iterations", "%d" % (iter), 
                            "--checkpoint_iterations", "%d" % (iter),])
            
            if (del_last_checkpoint):
                shutil.rmtree("output/whatever%d/" % (checkpoint-1))
            
            subprocess.run([sys.executable,
                            "render.py", 
                            "-m", "output/whatever%d" % checkpoint,
                            "--skip_train",])
            
            # subprocess.run([sys.executable,
            #                 "metrics.py", 
            #                 "-m", "output/whatever%d" % checkpoint])
            
            # with open("output/whatever%d/results.json" % checkpoint, "r") as file:
            #     data = json.load(file)

            # psnr = data["ours_%d" % (iter)]["PSNR"]

            # gaussian_count = plyextract.get_vertex_count("output/whatever%d/point_cloud/iteration_%d/point_cloud.ply" % (checkpoint, iter))

            # count_and_psnr[scene_dir].append((gaussian_count, psnr))

            checkpoint+=1

        print(f"{scene_dir}: count_and_psnr = {count_and_psnr[scene_dir]}")
        
        if (del_last_checkpoint):
            shutil.rmtree("output/whatever%d/" % (checkpoint-1))

gts = []
preds = []
heatmaps = []

for i, iter in enumerate(iterations):
    hm = visual_heatmap.Heatmap(chunk_size=8)
    pred_path = f'./output/whatever{i}/test/ours_{iter}/renders/00030.png'

    gt_path   = pred_path.replace('renders', 'gt')
    gt = np.array(Image.open(gt_path).convert('RGB'))
    gts.append(gt)

    pred = np.array(Image.open(pred_path).convert('RGB'))
    preds.append(pred)

    hm.create_heatmap(gt, pred)
    #hm.plot(pred_path.replace('renders/', 'heatmap_'))
    heatmaps.append(np.array(hm.heatmap))

fig, axes = plt.subplots(3, len(heatmaps))  

images = gts + preds + heatmaps

for i, iter in enumerate(iterations):
    axes[0, i].set_title(f"Iteration: {iter}")


for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='hot', interpolation='nearest')  
    ax.axis('off')  # turn off axis

output_path = "dodo"
fig.tight_layout()
#fig.subplots_adjust(wspace=0.0, hspace=0.0)
fig.savefig(output_path)
print(f"Heatmap saved to '{output_path}'")
fig.show()






