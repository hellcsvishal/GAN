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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 64 
NUM_EPOCHS = 50 # 50 is the sweet spot for a great demo
L1_LAMBDA = 100 

print(f"========== STARTING TRAINING ON: {DEVICE} ==========")

os.makedirs("evaluation", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 2. Data & Models
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = GANdataset(
    csv_file='/home/vishal-sharma/GAN/celebA/list_landmarks_align_celeba.csv', 
    root_dir='/home/vishal-sharma/GAN/celebA/img_align_celeba/img_align_celeba', 
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

gen = Generator().to(DEVICE)
disc = Discriminator().to(DEVICE)

BCE = nn.BCEWithLogitsLoss() 
L1_LOSS = nn.L1Loss()        

opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# --- THE SPEED HACK: AMP Scaler ---
scaler = torch.amp.GradScaler('cuda')

# 4. The Training Loop
for epoch in range(NUM_EPOCHS):
    for idx, (src_img, target_img, heatmap) in enumerate(loader):
        src_img = src_img.to(DEVICE)
        target_img = target_img.to(DEVICE)
        heatmap = heatmap.to(DEVICE)

        # --- THE BLINDFOLD TRICK (Crucial for Animation!) ---
        # Mask the bottom half of the source image so the generator CANNOT cheat
        src_img_masked = src_img.clone()
        src_img_masked[:, :, 70:, :] = -1 # -1 is pure black

        # --- STEP A: Train Discriminator ---
        opt_disc.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Generator gets the MASKED image and the heatmap
            fake_img = gen(src_img_masked, heatmap)
            
            disc_real = disc(src_img, target_img)
            loss_d_real = BCE(disc_real, torch.ones_like(disc_real))
            
            disc_fake = disc(src_img, fake_img.detach())
            loss_d_fake = BCE(disc_fake, torch.zeros_like(disc_fake))
            
            loss_disc = (loss_d_real + loss_d_fake) / 2

        scaler.scale(loss_disc).backward()
        scaler.step(opt_disc)
        scaler.update()

        # --- STEP B: Train Generator ---
        opt_gen.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            disc_fake_for_gen = disc(src_img, fake_img)
            loss_g_adv = BCE(disc_fake_for_gen, torch.ones_like(disc_fake_for_gen))
            
            # Loss is calculated against the UNMASKED target image!
            # It forces the AI to learn how to draw the missing mouth.
            loss_g_l1 = L1_LOSS(fake_img, target_img) * L1_LAMBDA
            
            loss_gen = loss_g_adv + loss_g_l1

        scaler.scale(loss_gen).backward()
        scaler.step(opt_gen)
        scaler.update()

        # 5. Monitoring Progress
        if idx % 100 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {idx} | Loss D: {loss_disc:.4f} Loss G: {loss_gen:.4f}")

    # 6. Save visual progress
    save_image(fake_img * 0.5 + 0.5, f"evaluation/gen_epoch_{epoch}.png")
    save_image(target_img * 0.5 + 0.5, f"evaluation/target_epoch_{epoch}.png")

    # 7. Save Model Checkpoints
    if epoch % 5 == 0:
        torch.save(gen.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
        print(f"=> Saved Checkpoint for Epoch {epoch}")
