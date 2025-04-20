import pickle
from PIL import Image
import numpy as np
import visual_heatmap
import matplotlib.pyplot as plt

name = "train"
with open(f"output/{name}.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
rendered_iterations = loaded_dict['rendered_iterations'] 
rendered_gaussian_counts = loaded_dict['rendered_gaussian_counts'] 

max_iter = rendered_iterations[-1]

mses = []
for i in range(37):

    pred_path = f'./output/{name}{max_iter}/test/ours_{max_iter}/renders/{str(i).zfill(5)}.png'

    gt_path   = pred_path.replace('renders', 'gt')
    gt = np.array(Image.open(gt_path).convert('RGB'))

    pred = np.array(Image.open(pred_path).convert('RGB'))

    me = np.mean((gt - pred)**2)
    mses.append(me)
    

best5 = np.argsort(mses)[:5][::-1] # Indices of 5 smallest numbers, in descending order

worst5 = np.argsort(mses)[-5:][::-1] # Indices of 5 largest numbers, in descending order



for angle in best5:
    gts = []
    preds = []
    heatmaps = []

    for i, iter in enumerate(rendered_iterations):
        hm = visual_heatmap.Heatmap(chunk_size=8)
        pred_path = f'./output/{name}{iter}/test/ours_{iter}/renders/{str(angle).zfill(5)}.png'

        gt_path   = pred_path.replace('renders', 'gt')
        gt = np.array(Image.open(gt_path).convert('RGB'))
        gts.append(gt)

        

        pred = np.array(Image.open(pred_path).convert('RGB'))
        preds.append(pred)

        me = np.mean((gt - pred)**2)
        #print(f"mse is {me} for iter {iter}")

        hm.create_heatmap(gt, pred)
        #hm.plot(pred_path.replace('renders/', 'heatmap_'))
        heatmaps.append(np.array(hm.heatmap))

    fig, axes = plt.subplots(3, len(heatmaps))  

    images = gts + preds + heatmaps

    for i, count in enumerate(rendered_gaussian_counts):
        axes[0, i].set_title(f"Gaussian_count: {count}", fontsize=10)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='hot', interpolation='nearest', vmin=0, vmax=160)  
        ax.axis('off')  # turn off axis

    
    output_path = f"dodo/best5/{name}_dodo_{angle}"
    
    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig(output_path)
    print(f"Heatmap saved to '{output_path}'")

for angle in worst5:
    gts = []
    preds = []
    heatmaps = []

    for i, iter in enumerate(rendered_iterations):
        hm = visual_heatmap.Heatmap(chunk_size=8)
        pred_path = f'./output/{name}{iter}/test/ours_{iter}/renders/{str(angle).zfill(5)}.png'

        gt_path   = pred_path.replace('renders', 'gt')
        gt = np.array(Image.open(gt_path).convert('RGB'))
        gts.append(gt)

        

        pred = np.array(Image.open(pred_path).convert('RGB'))
        preds.append(pred)

        me = np.mean((gt - pred)**2)
        #print(f"mse is {me} for iter {iter}")

        hm.create_heatmap(gt, pred)
        #hm.plot(pred_path.replace('renders/', 'heatmap_'))
        heatmaps.append(np.array(hm.heatmap))

    fig, axes = plt.subplots(3, len(heatmaps))  

    images = gts + preds + heatmaps

    for i, count in enumerate(rendered_gaussian_counts):
        axes[0, i].set_title(f"Gaussian_count: {count}")


    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='hot', interpolation='nearest', vmin=0, vmax=160)  
        ax.axis('off')  # turn off axis

    
    
    output_path = f"dodo/worst5/{name}_dodo_{angle}"
    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig(output_path)
    print(f"Heatmap saved to '{output_path}'")
        
