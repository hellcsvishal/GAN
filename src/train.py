import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from model import Generator, Discriminator
from dataset import GANdataset
import os

# 1. Configuration
DEVICE = "cpu"  # Specifically set to CPU
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
L1_LAMBDA = 100 

# Ensure a folder exists for your results
os.makedirs("evaluation", exist_ok=True)

# 2. Data & Models
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Make sure these paths match your local machine
dataset = GANdataset(
    csv_file='/home/vishal-sharma/GAN/celebA/list_landmarks_align_celeba.csv', 
    root_dir='/home/vishal-sharma/GAN/celebA/img_align_celeba/img_align_celeba', 
    transform=transform
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

# 3. Loss Functions & Optimizers
BCE = nn.BCEWithLogitsLoss() 
L1_LOSS = nn.L1Loss()        

opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# 4. The Training Loop
for epoch in range(NUM_EPOCHS):
    for idx, (src_img, target_img, heatmap) in enumerate(loader):
        # Move data to CPU
        src_img, target_img, heatmap = src_img.to(DEVICE), target_img.to(DEVICE), heatmap.to(DEVICE)

        # --- STEP A: Train Discriminator ---
        # The goal: Disc should say "1" for Real and "0" for Fake
        fake_img = gen(src_img, heatmap)
        
        # Judge the Real pair
        disc_real = disc(src_img, target_img)
        loss_d_real = BCE(disc_real, torch.ones_like(disc_real))
        
        # Judge the Fake pair (we .detach() fake_img so we don't backprop to Generator yet)
        disc_fake = disc(src_img, fake_img.detach())
        loss_d_fake = BCE(disc_fake, torch.zeros_like(disc_fake))
        
        loss_disc = (loss_d_real + loss_d_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # --- STEP B: Train Generator ---
        # The goal: Trick the Discriminator + Match the pixels of the target
        
        # 1. Adversarial Loss (How well did I trick the judge?)
        disc_fake_for_gen = disc(src_img, fake_img)
        loss_g_adv = BCE(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
        
        # 2. L1 Loss (Does the face actually look like the person?)
        loss_g_l1 = L1_LOSS(fake_img, target_img) * L1_LAMBDA
        
        loss_gen = loss_g_adv + loss_g_l1

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # 5. Monitoring Progress
        if idx % 10 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {idx} | Loss D: {loss_disc:.4f} Loss G: {loss_gen:.4f}")

    # 6. Save visual progress after every epoch
    # We "de-normalize" the image back to [0, 1] to save it properly
    save_image(fake_img * 0.5 + 0.5, f"evaluation/gen_epoch_{epoch}.png")
    save_image(src_img * 0.5 + 0.5, f"evaluation/input_epoch_{epoch}.png")

    # 7. Save Model Checkpoints
    if epoch % 5 == 0:
        torch.save(gen.state_dict(), "generator.pth")
        torch.save(disc.state_dict(), "discriminator.pth")
        print("=> Saved Checkpoints")
