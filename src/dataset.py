import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def generate_gaussian_heatmap(image_size, landmarks, sigma=3.0):
    """
    Creates a 1-channel heatmap with Gaussian 'glows' around landmarks.
    """
    h, w = image_size
    heatmap = np.zeros((1, h, w), dtype=np.float32) 
    Y, X = np.ogrid[0:h, 0:w]
    
    for (x, y) in landmarks:
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        dist_sq = (X - x)**2 + (Y - y)**2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        heatmap[0] = np.maximum(heatmap[0], gaussian)
        
    return torch.tensor(heatmap)

class GANdataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # FIXED: Using sep='\s+' to silence the Pandas warning
        self.landmarks_frame = pd.read_csv(csv_file, sep=',', header=1)
        self.landmarks_frame = self.landmarks_frame.iloc[:1000]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # FIXED: Pulling the string filename from the first column, just like your old code
        img_filename = str(self.landmarks_frame.iloc[idx, 0])
        img_name = os.path.join(self.root_dir, img_filename)
        image = Image.open(img_name).convert('RGB')
        orig_w, orig_h = image.size

        # FIXED: Shifting the columns to 1:11 since the filename is at 0
        landmarks = self.landmarks_frame.iloc[idx, 1:11].values.astype('float')
        landmarks = landmarks.reshape(-1, 2) 
        
        # Scale the coordinates
        scale_x = 128.0 / orig_w
        scale_y = 128.0 / orig_h
        scaled_points = [(int(x * scale_x), int(y * scale_y)) for x, y in landmarks]

        # Generate the soft heatmap
        heatmap_tensor = generate_gaussian_heatmap((128, 128), scaled_points, sigma=3.0)

        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = image

        return image_tensor, image_tensor, heatmap_tensor
