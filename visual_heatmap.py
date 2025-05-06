import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim


class Heatmap:
    # take two images (gt and pred) and create a heatmap of the mse between them
    def __init__(self, chunk_size=32, use_ssim=True):
        self.chunk_size = chunk_size
        self.heatmap = []
        self.use_ssim = use_ssim

    def create_heatmap(self, gt, pred):
        h, w = gt.shape[:2]
        ch, cw = self.chunk_size, self.chunk_size
        
        for y in range(0, h, ch):
            self.heatmap.append([])
            for x in range(0, w, cw):
                gt_chunk = gt[y:y+ch, x:x+cw]
                pred_chunk = pred[y:y+ch, x:x+cw]
                if self.use_ssim:
                    ssim_value = self.ssim(gt_chunk, pred_chunk)
                    self.heatmap[y//self.chunk_size].append(ssim_value)
                else:
                    mse_value = self.mse(gt_chunk, pred_chunk) 
                    self.heatmap[y//self.chunk_size].append(mse_value)

    def ssim(self, gt_block, pred_block):
        gt_block = gt_block.astype(np.float32)
        pred_block = pred_block.astype(np.float32)
        if gt_block.ndim == 3 and gt_block.shape[2] == 3:
            smallest_dim = min(gt_block.shape[0], gt_block.shape[1])
            win_size = min(self.chunk_size, smallest_dim)
            if win_size % 2 == 0: win_size -= 1
            if win_size <= 1: return 0
            data_range = max(1e-5, gt_block.max() - gt_block.min())
        return ssim(gt_block, pred_block, data_range=data_range, win_size=win_size, channel_axis=-1)
        
    def mse(self, gt_block, pred_block):
        return np.mean((gt_block - pred_block) ** 2)

    def plot(self, gt, pred, output_path='./heatmap.png'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        lim = (-1, 1) if self.use_ssim else (0, 160)

        titles = ['Ground Truth', 'Prediction', 'Heatmap']
        images = [gt, pred, self.heatmap]

        im = None
        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, cmap='hot', interpolation='nearest', vmin=lim[0], vmax=lim[1])
            ax.set_title(title)
            ax.axis('off')
        
        # cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, orientation='horizontal')
        # cbar.set_label('SSIM' if self.use_ssim else 'MSE', labelpad=15)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Heatmap saved to '{output_path}'")
        

def main():
    # Run 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1'
    # Before running this script
    pred_path = 'output/bicycle2100/test/ours_2100/renders/00000.png'
    gt_path   = pred_path.replace('renders', 'gt')
    gt = np.array(Image.open(gt_path).convert('RGB'))
    pred = np.array(Image.open(pred_path).convert('RGB'))

    hm = Heatmap(chunk_size=3)
    hm.create_heatmap(gt, pred)
    hm.plot(gt, pred, pred_path.replace('renders/', f'heatmap{hm.chunk_size}x{hm.chunk_size}_'))

if __name__ == "__main__":
    main()