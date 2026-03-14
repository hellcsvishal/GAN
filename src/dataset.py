import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class GANdataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        """
        Args:

            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied.
        """
        self.landmarks=pd.read_csv(csv_file,nrows=1000)
        self.root_dir=root_dir
        self.transform=transform


    def __len__(self):

        return len(self.landmarks)

   
    def __getitem__(self,idx):
        # 1. Get path and label from your lists
        # 1.1 Get the image name from the dataframe at 'index'
        image_name=self.landmarks.iloc[idx, 0]
        # construct full path
        img_path=os.path.join(self.root_dir,image_name)


        # 2. Open image (remember the RGB trick!)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size        
        # 3. Transform
        if self.transform:
            image = self.transform(image)

        _, new_h, new_w = image.shape    

        landmarks = self.landmarks.iloc[idx, 1:].values.astype("float")
        landmarks = landmarks.reshape(-1, 2)

        landmarks[:, 0] = landmarks[:, 0] * (new_w / orig_w) # Scale X
        landmarks[:, 1] = landmarks[:, 1] * (new_h / orig_h) # Scale Y

        # Create the blank mask
        # Create the blank mask
        heatmap = torch.zeros((1, new_h, new_w))

      # Loop through each of the 5 scaled landmarks
        for point in landmarks:
           x, y = int(point[0]), int(point[1])
    
      # Check if the point is within the image bounds
           if 0 <= x < new_w and 0 <= y < new_h:
        # Set the pixel to 1.0
              heatmap[0, y, x] = 1.0# Create the blank mask
        # Return: (Image to be processed, The Goal Image, The Landmark Mask)
        return image, image, heatmap    


import matplotlib.pyplot as plt

if __name__ == "__main__":
    from torchvision import transforms
    
    # 1. Define your transform (standard 128x128 for GANs)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Standard GAN normalization
    ])

    # 2. Initialize your dataset with your paths
    dataset = GANdataset(
        csv_file='/home/vishal-sharma/GAN/celebA/list_landmarks_align_celeba.csv', 
        root_dir='/home/vishal-sharma/GAN/celebA/img_align_celeba/img_align_celeba', 
        transform=transform
    )

    # 3. Get the first sample
    img, heat = dataset[0]

    # 4. Visualization Logic
    # We need to de-normalize the image from [-1, 1] to [0, 1] to display it
    img_display = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    heat_display = heat.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Transformed Image")
    plt.imshow(img_display)
    
    plt.subplot(1, 2, 2)
    plt.title("Anatomical Heatmap")
    plt.imshow(heat_display, cmap='hot')
    plt.savefig('test_output.png')
    print("Image saved as test_output.png")
    



