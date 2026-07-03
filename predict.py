# Download the pretrained model weights
# and place them inside the models folder

#---------------------------------------
# Brain Tumor Segmentation Prediction
# U-Net + ResNet34
#---------------------------------------

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models
from torchvision import transforms

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"D:\AI Medical Imaging\Project\Brain-Tumor-Segmentation-U-Net\models\best_model_with_ResNet34.pth"
IMAGE_PATH = r"D:\AI Medical Imaging\Project\LGG MRI Segmentation Dataset\kaggle_3m\TCGA_CS_4941_19960909\TCGA_CS_4941_19960909_13.tif"
IMAGE_SIZE = 256
THRESHOLD = 0.5

# Double Convolution Block
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv(x)
    
# Decoder Block
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            out_channels * 2,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x  

# U-Net with ResNet34 Encoder
class ResNet34_UNet(nn.Module):

    def __init__(self, out_channels=1):

        super().__init__()

        resnet = models.resnet34(weights=None)

        # Encoder

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )

        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Bottleneck

        self.bottleneck = nn.Sequential(
            DoubleConv(512,1024),
            nn.Dropout2d(0.3)
        )

        # Decoder

        self.decoder1 = DecoderBlock(1024,256)
        self.decoder2 = DecoderBlock(256,128)
        self.decoder3 = DecoderBlock(128,64)

        # Final Upsampling

        self.up_final = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        self.up_final2 = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        # Final Layer

        self.final_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self,x):

        # Encoder

        skip1 = self.initial(x)

        x = self.maxpool(skip1)

        skip2 = self.encoder1(x)
        skip3 = self.encoder2(skip2)
        skip4 = self.encoder3(skip3)

        x = self.encoder4(skip4)

        # Bottleneck

        x = self.bottleneck(x)

        # Decoder

        x = self.decoder1(x,skip4)
        x = self.decoder2(x,skip3)
        x = self.decoder3(x,skip2)

        # Recover Original Resolution

        x = self.up_final(x)
        x = self.up_final2(x)
        x = self.final_conv(x)

        return x

# Load Model
model = ResNet34_UNet().to(DEVICE)
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE,IMAGE_SIZE)
    ),

    transforms.ToTensor()
])

# Load MRI Image
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

# Prediction

with torch.no_grad():
    pred = model(input_tensor)
    pred = torch.sigmoid(pred)
    pred = (pred > THRESHOLD).float()
prediction = pred.squeeze().cpu().numpy()

# Tumor Statistics
tumor_pixels = np.sum(prediction)
total_pixels = prediction.shape[0] * prediction.shape[1]
tumor_percentage = (tumor_pixels / total_pixels) * 100

# Find Tumor Contours
prediction_uint8 = (prediction * 255).astype(np.uint8)

contours, _ = cv2.findContours(
    prediction_uint8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

prediction_rgb = cv2.cvtColor(
    prediction_uint8,
    cv2.COLOR_GRAY2RGB
)

cv2.drawContours(
    prediction_rgb,
    contours,
    -1,
    (0,255,0),
    2
)

# Tumor Centroid
if len(contours) > 0:

    largest = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest)

    if M["m00"] != 0:

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(
            prediction_rgb,
            (cx, cy),
            5,
            (255, 0, 0),
            -1
        )

# Display Results
image_display = image.resize((IMAGE_SIZE, IMAGE_SIZE))
plt.figure(figsize=(10,5))

# MRI
plt.subplot(1,2,1)
plt.imshow(image_display)
plt.title("MRI Image")
plt.axis("off")

# Prediction
plt.subplot(1,2,2)
plt.imshow(prediction_rgb)
plt.text(
    5,
    20,
    f"Tumor Area : {tumor_pixels:.0f} pixels\n"
    f"Tumor Ratio : {tumor_percentage:.2f}%",
    fontsize=10,
    color="red",
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.title("Prediction")
plt.axis("off")
plt.tight_layout()
plt.show()