import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import pandas as pd
from matplotlib import pyplot as plt

# Define the Super-Resolution CNN model for grayscale images

output_folder = "./upsampled_images/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bicubic', align_corners=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Load grayscale data from CSV
def load_images_from_csv(filename, size):
    df = pd.read_csv(filename)

    # Set the "Id" column as the index
    df.index = df.pop("Id")

    # Reshape the data to form the grayscale images
    images = df.to_numpy().reshape(-1, 1, size, size)

    # Convert to tensors
    tensor_images = [torch.tensor(image, dtype=torch.float32)
                     for image in images]

    return tensor_images


low_res_images = load_images_from_csv("train_small.csv", 50)
high_res_images = load_images_from_csv("train_big.csv", 100)

dataset = TensorDataset(torch.stack(low_res_images),
                        torch.stack(high_res_images))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training the model
model = SRCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for low_res, high_res in dataloader:
        outputs = model(low_res)
        loss = criterion(outputs, high_res)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), 'srcnn_model_grayscale.pth')

# To upsample a new 50x50 grayscale image loaded from another CSV
model.eval()
with torch.no_grad():
    new_low_res_images = load_images_from_csv("test_small.csv", 50)
    new_high_res_images = load_images_from_csv("test_big.csv", 100)
    for idx, new_low_res_image in enumerate(new_low_res_images):
        upsampled_image = model(new_low_res_image.unsqueeze(0)).squeeze(0)
        upsampled_image_np = upsampled_image.numpy().squeeze()

        # Ensure the images from CSVs are 2D as well
        original_low_res_np = new_low_res_image.numpy().squeeze()
        original_high_res_np = new_high_res_images[idx].numpy().squeeze()

        plt.imsave(os.path.join(output_folder,
                   f"{idx}_upsampled.png"), upsampled_image_np, cmap="copper")
        plt.imsave(os.path.join(output_folder,
                   f"{idx}_big.png"), original_high_res_np, cmap="copper")
        plt.imsave(os.path.join(output_folder,
                   f"{idx}_small.png"), original_low_res_np, cmap="copper")
