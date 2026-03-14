import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(DownBlock, self).__init__()
        
        # We use a list to store our layers
        layers = []
        
        # 1. Convolution: This is what actually 'sees' the features.
        layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
        
        # 2. Batch Normalization: This keeps the math stable so the GAN doesn't 'explode'.
        # We skip this only on the very first layer of the Generator.
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        # 3. Activation: LeakyReLU allows a tiny bit of negative signal to pass through,
        # which helps GANs learn much better than standard ReLU.
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Wrap all these into a single sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, use_dropout=False):
        super(UpBlock, self).__init__()
        layers = []
        
        # 1. Transposed Conv: Doubles the H and W (Stride 2)
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
        
        # 2. Batch Norm
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        # 3. Dropout (Only if requested)
        if use_dropout:
            layers.append(nn.Dropout(0.5))
            
        # 4. Activation (Standard ReLU for Decoder)
        layers.append(nn.ReLU(inplace=True))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = DownBlock(4, 64, use_bn=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)

        # Decoder (Going back up)
        # Input to up1 is d7 (512). Output is 512.
        self.up1 = UpBlock(512, 512, use_dropout=True) 
        
        # Input to up2 is (up1_output + d6). Since up1_output is 512 and d6 is 512, 
        # the next in_channels will be 1024!
        self.up2 = UpBlock(1024, 512, use_dropout=True)
        self.up3 = UpBlock(1024, 512, use_dropout=True)
        self.up4 = UpBlock(1024, 256)
        self.up5 = UpBlock(512, 128)
        self.up6 = UpBlock(256, 64)
        
        # Final Layer: To get back to 3-channel RGB image
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh() # Tanh forces pixels to be between [-1, 1]
        )


    def forward(self,x,heatmap):
        # 1. Glue image and heatmap together
        # x is (Batch, 3, 128, 128), heatmap is (Batch, 1, 128, 128)
        d0 = torch.cat([x, heatmap], dim=1) # Notice dim=1 for channels
        
        # 2. Pass through layers and save the results
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
         # 3. Upsampling with Skip Connections
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], dim=1)) # Skip connection!
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        
        # Final output
        output = self.final_up(torch.cat([u6, d1], dim=1))
        
        return output

class Discriminator(nn.Module):
    def __init__(self,in_channels=3):
        super(Discriminator,self).__init__()
        self.initial = DownBlock(in_channels * 2, 64, use_bn=False)
        self.model=nn.Sequential(
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            # The final layer: we want a 1-channel 'heatmap' of realness
            # We don't use DownBlock here because we don't want to shrink it by 2
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )
    def forward(self, x, y):
        # Concatenate Source image 'x' and Target image 'y'
        # Shape: [Batch, 6, 128, 128]
        input_data = torch.cat([x, y], dim=1)
        return self.model(self.initial(input_data))    

if __name__ == "__main__":
    # Test Data
    fake_img = torch.randn((1, 3, 128, 128))
    fake_heat = torch.randn((1, 1, 128, 128))
    
    # 1. Test Generator
    gen = Generator()
    gen.eval()
    generated_face = gen(fake_img, fake_heat)
    print(f"Gen Output: {generated_face.shape}")
    
    # 2. Test Discriminator
    disc = Discriminator()
    # Discriminator compares Source Image and Generated Image
    prediction = disc(fake_img, generated_face)
    print(f"Disc Prediction Grid Shape: {prediction.shape}")
