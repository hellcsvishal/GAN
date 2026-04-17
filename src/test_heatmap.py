from torchvision.utils import save_image
from dataset import GANdataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Using your paths
dataset = GANdataset(
    csv_file='/home/vishal-sharma/GAN/celebA/list_landmarks_align_celeba.csv', 
    root_dir='/home/vishal-sharma/GAN/celebA/img_align_celeba/img_align_celeba', 
    transform=transform
)

src_img, target_img, heatmap = dataset[0]

save_image(heatmap, "test_gaussian_glow.png")
save_image(src_img * 0.5 + 0.5, "test_source_face.png")
print("Saved! Check test_gaussian_glow.png")
