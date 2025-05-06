import os
import pickle
from PIL import Image
import numpy as np
import visual_heatmap
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from visual_heatmap import Heatmap

name = "truck"

with open(f"output/{name}.pkl", "rb") as file:
    loaded_dict = pickle.load(file)
rendered_iterations = loaded_dict['rendered_iterations'] 
rendered_gaussian_counts = loaded_dict['rendered_gaussian_counts'] 

max_iter = rendered_iterations[-1]

test_set = os.listdir(f'./output/{name}{max_iter}/test/ours_{max_iter}/renders')
print(f'Evaluating {len(test_set)} images')
mses = []
for img in test_set:

    pred_path = f'./output/{name}{max_iter}/test/ours_{max_iter}/renders/{img}'

    gt_path = pred_path.replace('renders', 'gt')
    gt = np.array(Image.open(gt_path).convert('RGB'))
    pred = np.array(Image.open(pred_path).convert('RGB'))

    # me = np.mean((gt - pred)**2)
    me = ssim(gt, pred, channel_axis=-1)
    mses.append(me)


best5 = np.argsort(mses)[:5][::-1] # Indices of 5 smallest numbers, in descending order

worst5 = np.argsort(mses)[-5:][::-1] # Indices of 5 largest numbers, in descending order


def generate_heatmaps(angle, rendered_iterations, name, use_ssim=False):
    gts, preds, heatmaps = [], [], []

    for i, iter in enumerate(rendered_iterations):
        hm = visual_heatmap.Heatmap(chunk_size=3, use_ssim=use_ssim)
        pred_path = f'./output/{name}{iter}/test/ours_{iter}/renders/{str(angle).zfill(5)}.png'
        gt_path   = pred_path.replace('renders', 'gt')

        gt = np.array(Image.open(gt_path).convert('RGB'))
        gts.append(gt)

        pred = np.array(Image.open(pred_path).convert('RGB'))
        preds.append(pred)

        if use_ssim:
            me = ssim(gt, pred, channel_axis=-1)
        else:
            me = np.mean((gt - pred)**2)
        #print(f"mse is {me} for iter {iter}")

        hm.create_heatmap(gt, pred)
        #hm.plot(pred_path.replace('renders/', 'heatmap_'))
        heatmaps.append(np.array(hm.heatmap))

    return gts, preds, heatmaps

def plot_heatmaps(angle, gts, preds, heatmaps, rendered_gaussian_counts, output_path, is_ssim=False):
    fig, axes = plt.subplots(3, len(heatmaps))  

    images = gts + preds + heatmaps

    for i, count in enumerate(rendered_gaussian_counts):
        axes[0, i].set_title(f"Gaussian count: {count}", fontsize=10)
        me = ssim(gts[i], preds[i], channel_axis=-1) if is_ssim else np.mean((gts[i] - preds[i])**2)
        axes[2, i].set_title(f'Avg. {"SSIM" if is_ssim else "MSE"}: {me:.3f}', fontsize=10)
    lim = (-1, 1) if is_ssim else (0, 160)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='hot', interpolation='nearest', vmin=lim[0], vmax=lim[1])  
        ax.axis('off')  # turn off axis

    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig(output_path)
    print(f"Heatmap saved to '{output_path}'")

def process_angles(angles, name, rendered_iterations, rendered_gaussian_counts, prefix, ssim=False):
    for angle in angles:
        gts, preds, heatmaps = generate_heatmaps(angle, rendered_iterations, name, use_ssim=ssim)
        output_path = f"output/{prefix}_{name}_dodo_{angle}"
        plot_heatmaps(angle, gts, preds, heatmaps, rendered_gaussian_counts, output_path, is_ssim=ssim)

print('Creating heatmaps for 5 best angles (using SSIM may take a while)...')
process_angles(best5, name, rendered_iterations, rendered_gaussian_counts, "best", ssim=True)
print('Creating heatmaps for 5 worst angles (using SSIM may take a while)...')
process_angles(worst5, name, rendered_iterations, rendered_gaussian_counts, "worst", ssim=True)