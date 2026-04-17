import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import Generator
from dataset import generate_gaussian_heatmap

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Puppeteer on {DEVICE}...")

gen = Generator().to(DEVICE)

try:
    # Load the raw dictionary
    raw_state_dict = torch.load("generator.pth", map_location=DEVICE)
    
    # Clean the keys (Removes the PyTorch 2.0 '_orig_mod.' prefix)
    clean_state_dict = {}
    for key, value in raw_state_dict.items():
        clean_key = key.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = value
        
    # Load the cleaned dictionary into the model
    gen.load_state_dict(clean_state_dict)
    print("=> Generator loaded successfully. Ready to animate!")
except FileNotFoundError:
    print("=> WARNING: generator.pth not found. Please ensure the file is in the same directory.")
    exit()

gen.eval()

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Make sure you have a test image ready in your folder!
# You can change this path to any image from the CelebA dataset
image_path = "test_source_face.png" 
try:
    pil_img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
except FileNotFoundError:
    print(f"Error: Could not find '{image_path}'. Please provide a valid image.")
    exit()

# ==========================================
# 3. INTERACTIVE UI SETUP
# ==========================================
# Default Landmarks: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
landmarks = [(45, 55), (83, 55), (64, 80), (45, 100), (83, 100)]
selected_point = -1 

def mouse_callback(event, x, y, flags, param):
    global landmarks, selected_point
    
    # Scale mouse coordinates down by 3 (because the window is 3x zoomed)
    real_x = x // 3
    real_y = y // 3
    
    # Click Down: Grab the closest landmark
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (lx, ly) in enumerate(landmarks):
            if (lx - real_x)**2 + (ly - real_y)**2 < 64: 
                selected_point = i
                break
                
    # Drag: Move the selected landmark
    elif event == cv2.EVENT_MOUSEMOVE and selected_point != -1:
        new_x = max(0, min(127, real_x))
        new_y = max(0, min(127, real_y))
        landmarks[selected_point] = (new_x, new_y)
        
    # Let Go: Release the landmark
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = -1

# ==========================================
# 4. THE LIVE RENDER LOOP
# ==========================================
cv2.namedWindow("FGAN Puppeteer (Click and Drag!)")
cv2.setMouseCallback("FGAN Puppeteer (Click and Drag!)", mouse_callback)

print("\n=> PUPPETEER ONLINE!")
print("=> Instructions: Click and drag the green dots on the mouth. Press 'q' in the window to quit.")

while True:
    # --- A. Draw the UI Window ---
    ui_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    for (x, y) in landmarks:
        cv2.circle(ui_img, (x, y), 3, (0, 255, 0), -1)

    # --- B. Run the AI ---
    with torch.no_grad():
        heatmap = generate_gaussian_heatmap((128, 128), landmarks, sigma=3.0)
        heatmap = heatmap.unsqueeze(0).to(DEVICE)
        
        # THE BLINDFOLD TRICK: Black out the lower half of the face 
        # so the AI has to draw a new mouth based on the heatmap
        masked_img = img_tensor.clone()
        masked_img[:, :, 70:, :] = -1 
        
        # Feed masked image + heatmap to generator
        fake_img = gen(masked_img, heatmap)

    # --- C. Format the Output for Screen ---
    output_tensor = fake_img.squeeze(0).cpu() * 0.5 + 0.5
    output_np = output_tensor.numpy().transpose(1, 2, 0)
    output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # --- D. Display the Results ---
    combined_view = np.hstack((ui_img, output_bgr))
    
    # Scale up by 3x so it fills your screen nicely
    display_view = cv2.resize(combined_view, (128 * 2 * 3, 128 * 3), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("FGAN Puppeteer (Click and Drag!)", display_view)

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
