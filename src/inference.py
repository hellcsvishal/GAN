import torch 
from torchvision import transforms
from torchvision.utils import save_image
from model import Generator
from PIL import Image
import numpy as np

DIVICE ="cuda" if torch.cuda.is_available() else "cpu"

def load_model(checkpoint_path):
    model=Generator().to(DEVICE)
    # map_location ensures it works even if you trained on GPU but run on CPU
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transorms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
def run_inference(model,source_image_path, heatmap_tensor):
    img=Image.open(source_image_path).convert("RGB")
    img_tensor=transform(img).unsqueeze(0).to(DEVICE)

    if heatmap_tensor.ndimension()==3:
        heatmap_tensor=heatmap_tensor.unsqueeze(0)

        with torch.no_grad():
            generated_img=model(img_tensor,heatmap_tensor)


        result=generated_img.squeeze(0)* 0.5 + 0.5
        save_image(result,"animation_result.png")
        print("result saved as animation_result.png")

if __name__=="__main__":
    pass
