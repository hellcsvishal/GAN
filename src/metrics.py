import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from model import Generator
import tqdm
from dataset import GANdataset

# 1. Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the Model
gen = torch.compile(Generator().to(DEVICE), backend="inductor")
try:
    gen.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    print("=> Loaded generator.pth successfully.")
except FileNotFoundError:
    print("=> Warning: generator.pth not found! Running with random weights just to test the script.")
gen.eval() # Set to evaluation mode!

# 3. Data Setup (Using your exact file paths)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = GANdataset(
    csv_file='celebA/list_landmarks_align_celeba.csv', 
    root_dir='celebA/img_align_celeba/img_align_celeba', 
    transform=transform
)

# We only need a small batch to get a statistically significant average
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Metric Setup
# SSIM expects image values between 0.0 and 1.0
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
ssim_scores = []

print(f"Calculating SSIM over {len(loader)} images...\n")

with torch.no_grad():
    loop = tqdm.tqdm(loader)
    for i, (src_img, target_img, heatmap) in enumerate(loop):
            
        src_img = src_img.to(DEVICE)
        heatmap = heatmap.to(DEVICE)

        # Generate the animated face
        fake_img = gen(src_img, heatmap)

        # Un-normalize from [-1, 1] back to [0, 1] for the SSIM formula
        src_img_norm = src_img * 0.5 + 0.5
        fake_img_norm = fake_img * 0.5 + 0.5

        # Calculate SSIM between the Original Face and the Generated Face
        score = ssim(fake_img_norm, src_img_norm)
        ssim_scores.append(score.item())

# 5. Print the Final Report
average_ssim = np.mean(ssim_scores)

print("="*50)
print(" 📊 FGAN SCIENTIFIC BENCHMARK REPORT")
print("="*50)
print(f"Metric Evaluated   : Structural Similarity Index (SSIM)")
print(f"Data Source        : CelebA (128x128)")
print(f"SSIM Score : {average_ssim}")
print("="*50)

if average_ssim > 0.85:
    print("Conclusion: EXCELLENT Identity Preservation.")
elif average_ssim > 0.60:
    print("Conclusion: Fair preservation. Model is still learning.")
else:
    print("Conclusion: Low similarity. Waiting for more epochs.")
