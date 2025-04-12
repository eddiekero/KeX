import sys
import subprocess
import json
import plyextract
import shutil


checkpoint = 0
iteration_step = 100
max_iteration = 500

count_and_psnr = []


subprocess.run([sys.executable,
                    "train.py", 
                    "--model_path", "output/whatever0",
                    "-s", "tandt_db\\tandt\\truck\\",
                    "--eval",
                    "--iterations", "100", 
                    "--checkpoint_iterations", "100",])
checkpoint+=1

for iter in range(100, max_iteration, iteration_step):
    subprocess.run([sys.executable,
                    "train.py", 
                    "--model_path", "output/whatever%d" % (checkpoint),
                    "-s", "tandt_db\\tandt\\truck\\",
                    "--eval",
                    "--start_checkpoint", "output/whatever%d/chkpnt%d.pth" % (checkpoint-1, iter),
                    "--iterations", "%d" % (iter+100), 
                    "--checkpoint_iterations", "%d" % (iter+100),])
    
    shutil.rmtree("output/whatever%d/" % (checkpoint-1))
    
    subprocess.run([sys.executable,
                    "render.py", 
                    "-m", "output/whatever%d" % checkpoint,
                    "--skip_train",])
    
    subprocess.run([sys.executable,
                    "metrics.py", 
                    "-m", "output/whatever%d" % checkpoint,])
    
    with open("output/whatever%d/results.json" % checkpoint, "r") as file:
        data = json.load(file)

    psnr = data["ours_%d" % (iter+100)]["PSNR"]

    gaussian_count = plyextract.get_vertex_count("output/whatever%d/point_cloud/iteration_%d/point_cloud.ply" % (checkpoint, iter+100))

    count_and_psnr.append((gaussian_count, psnr))

    checkpoint+=1

print(count_and_psnr)