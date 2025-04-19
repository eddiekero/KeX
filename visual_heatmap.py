import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


class Heatmap:
    # take two images (gt and pred) and create a heatmap of the mse between them
    def __init__(self, chunk_size=32):
        self.chunk_size = chunk_size
        self.heatmap = []

    def create_heatmap(self, gt, pred): 
        h, w = gt.shape[:2]
        ch, cw = self.chunk_size, self.chunk_size
        
        for y in range(0, h, ch):
            self.heatmap.append([])
            for x in range(0, w, cw):
                gt_chunk = gt[y:y+ch, x:x+cw]
                pred_chunk = pred[y:y+ch, x:x+cw]
                mse_value = self.mse(gt_chunk, pred_chunk)
                self.heatmap[y//self.chunk_size].append(mse_value)
        
        
    def mse(self, gt_block, pred_block):
        return ((gt_block - pred_block) ** 2).mean()

    def plot(self, output_path='./heatmap.png'):
        plt.imshow(np.array(self.heatmap), cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label('Mean Squared Error', rotation=270, labelpad=15)
        plt.title(f'Reconstruciton Accuracy Heatmap ({self.chunk_size}x{self.chunk_size} chunks)')
        plt.savefig(output_path)
        print(f"Heatmap saved to '{output_path}'")

def main():
    # Run 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1'
    # Before running this script
    pred_path = 'output/whatever1/test/ours_2500/renders/00030.png'
    gt_path   = pred_path.replace('renders', 'gt')
    gt = np.array(Image.open(gt_path).convert('RGB'))
    pred = np.array(Image.open(pred_path).convert('RGB'))

    hm = Heatmap(chunk_size=8)
    hm.create_heatmap(gt, pred)
    hm.plot(pred_path.replace('renders/', 'heatmap_'))

if __name__ == "__main__":
    main()