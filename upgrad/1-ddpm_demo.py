# Env creation
#  python -m venv myvenv
#  source myvenv/bin/activate
#  Windows - activate myvenv\Scripts\activate
# Packages - pip install torch torchvision matplotlib numpy
# 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import time

# Create output directory for generated images
os.makedirs("ddpm_output", exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 28
batch_size = 64
epochs = 20
timesteps = 1000  # Number of steps in the diffusion process
lr = 1e-4

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist_data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t):
        return self.net(x)

# Forward diffusion process (add noise)
def forward_diffusion(x, t, betas):
    noise = torch.randn_like(x).to(device)
    alpha_t = torch.cumprod(1 - betas, dim=0)[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
    x_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise
    return x_t, noise

# Linear beta schedule
def beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps).to(device)

# Training the Diffusion Model
diffusion_model = DiffusionModel().to(device)
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
criterion = nn.MSELoss()

betas = beta_schedule(timesteps)
for epoch in range(epochs):
    diffusion_model.train()
    train_loss = 0
    start_time = time.time()

    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        optimizer.zero_grad()

        # Random timestep
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device).long()

        # Forward diffusion
        x_t, noise = forward_diffusion(imgs, t, betas)

        # Predict noise
        noise_pred = diffusion_model(x_t, t.view(-1, 1, 1, 1))  # Reshape t for broadcasting
        loss = criterion(noise_pred, noise)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Log progress for every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

    end_time = time.time()
    print(f"Epoch [{epoch+1}/{epochs}] completed in {end_time - start_time:.2f}s with Loss: {train_loss / len(dataloader):.4f}")

    # Generate images at the end of each epoch
    with torch.no_grad():
        diffusion_model.eval()
        sample = torch.randn((batch_size, 1, image_size, image_size), device=device)  # Start from noise
        print(f"Generating images for epoch {epoch+1}...")
        for t in reversed(range(timesteps)):
            alpha_t = torch.cumprod(1 - betas, dim=0)[t]
            noise_pred = diffusion_model(sample, t.view(-1, 1, 1, 1))  # Reshape t for broadcasting
            sample = (sample - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        save_image(sample, f"ddpm_output/generated_epoch_{epoch+1}.png", nrow=8, normalize=True)

print("Training complete. Check the 'ddpm_output' folder for generated images.")