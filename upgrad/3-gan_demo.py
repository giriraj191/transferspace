# Create Env
# Packages - pip install torch torchvision matplotlib numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
os.makedirs("gan_output", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_size = 28
batch_size = 64
epochs = 50
lr = 0.0002
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to range [-1, 1]
])
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST("mnist_data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, img_size * img_size),
            nn.Tanh()  # Output normalized between [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, img_size, img_size)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real or fake
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Ground truths
        valid = torch.ones(imgs.size(0), 1, device=device)
        fake = torch.zeros(imgs.size(0), 1, device=device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        generated_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(generated_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(imgs.to(device)), valid)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print training progress
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save generated images at the end of each epoch
    save_image(generated_imgs.data[:25], f"gan_output/{epoch}.png", nrow=5, normalize=True)




example_imgs = os.listdir("gan_output")[-1]
plt.imshow(plt.imread(f"gan_output/{example_imgs}"))
plt.title("Generated Images")
plt.axis("off")
plt.show()
