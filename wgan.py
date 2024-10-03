# wgan.py

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable, grad

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def gradient_penalty(D, real_samples, fake_samples, device='cpu'):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.requires_grad_(True)
    
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

def prepare_wgan_data(X):
    tensor_X = torch.FloatTensor(X)
    return torch.utils.data.DataLoader(tensor_X, batch_size=64, shuffle=True)

def train_wgan(data_loader, device='cpu', latent_dim=100, hidden_dim=128, lr=1e-4, lambda_gp=10, n_epochs=100, n_critic=5):
    G = Generator(latent_dim, hidden_dim, data_loader.dataset.shape[1]).to(device)
    D = Discriminator(data_loader.dataset.shape[1], hidden_dim).to(device)
    
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(n_epochs):
        for i, real_samples in enumerate(data_loader):
            real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)
            
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_samples = G(z)
                
                D_real = D(real_samples)
                D_fake = D(fake_samples)
                
                gp = gradient_penalty(D, real_samples, fake_samples, device)
                loss_D = torch.mean(D_fake) - torch.mean(D_real) + lambda_gp * gp
                
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = G(z)
            loss_G = -torch.mean(D(fake_samples))
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        
        print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")
    
    return G, D