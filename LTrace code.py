# Import relevant libraries
import h5py                             # Read the images files
import random                           # Generate random values
import torch                            # Core library for PyTorch
import numpy as np                      # Better handle the data 
import torch.nn as nn                   # Building neural network layers and loss functions
import torch.optim as optim             # Provides optimization algorithms
from torchvision import transforms      # Provides efficient trasnformer functions
from torch.utils.data import DataLoader # Create data loaders
import torch.nn.functional as F         # Apply convolutions
import matplotlib.pyplot as plt         # Draw plots to visualize the results     
from tqdm import tqdm                   # Show progress

# Specify the directory of the file and set it to the variable "file_dir"
file_dir = r"C:\Users\andre\Desktop\VAE\train_images.h5"

# Load the "train_images.h5" file that containst the images
with h5py.File(file_dir, "r") as file:
    data = file['train_images'][:]

# Check the data type
type(data)

# Check its shape
data.shape

# Correct to default tensor doformat [50000, 1, 128, 128]
data = np.moveaxis(data, -1, 1) 
print(f'New shape: {data.shape}') 

# Convert the data into a tensor
tensor_data = torch.tensor(data, dtype=torch.float32)

# Devide the data into batches for faster run times
batch_size = 32# The number examples utilized in one iteration of model training.
data_loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)

# Specify the latent (z) dimension size
z_dim = 16

# MODEL 1:  Implement a convoluional + dense (fully conected) layer VAE
class ConvDenseVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDenseVAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # Output: 32x64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Output: 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: 128x16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 512) # Flatten and output size: 512
        self.fc_mu = nn.Linear(512, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(512, latent_dim) # Log variance of latent space
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 512) # Input size: 16 (latent space), output size: 512
        self.fc3 = nn.Linear(512, 128 * 16 * 16) # Inout size = 512, Output size:128*16*16
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # Output size: 64x32x32
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # Output size: 32x64x64
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # Output size: 1x128x128
    
    def encode(self, x):
        h = F.relu(self.conv1(x))  # Apply the first convolutional layer
        h = F.relu(self.conv2(h))  # Apply the second convolutional layer
        h = F.relu(self.conv3(h))  # Apply the third convolutional layer
        h = h.view(h.size(0), -1)  # Flatten the output from the convolutional layers
        h = F.relu(self.fc1(h))    # Apply the first fully connected layer (FC layer)
        mu = self.fc_mu(h)         # Compute the mean of the latent space distribution
        logvar = self.fc_logvar(h) # Compute the log variance of the latent space distribution
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Compute the standard deviation from the log variance
        eps = torch.randn_like(std) # Sample epsilon from a standard normal distribution with the same shape as std
        return mu + eps * std       # Generate samples from the latent space

    def decode(self, z):
        h = F.relu(self.fc2(z))     # Apply the first fully connected layer
        h = F.relu(self.fc3(h))     # Apply the second fully connected layer
        h = h.view(h.size(0), 128, 16, 16)  # Reshape the output from the fully connected layers into a 4D tensor
        h = F.relu(self.deconv1(h)) # Apply the first transposed convolution layer
        h = F.relu(self.deconv2(h)) # Apply the second transposed convolution layer
        return torch.sigmoid(self.deconv3(h))     # Apply the third transposed convolution layer and use sigmoid activation to get final output


    def forward(self, x):
        mu, logvar = self.encode(x)     # Encode the input data to obtain the mean and log variance of the latent space distribution
        z = self.reparameterize(mu, logvar) # Reparameterize to obtain latent vectors from the mean and log variance
        return self.decode(z), mu, logvar # Decode the latent vectors to reconstruct the input data
    
# Compute the loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine the device as CUDA if available, otherwise CPU
model1 = ConvDenseVAE(latent_dim=z_dim).to(device) # Initialize the VAE model and move it to the selected device
optimizer = optim.Adam(model.parameters(), lr = 1e-4) # Define the optimizer
loss_fn = nn.BCELoss(reduction = "sum") # Define the loss function to measure reconstruction loss using Binary Cross Entropy (BCE)
num_epochs = 10 # Define the number of epochs for training

# Training loop for the specified number of epochs
for epoch in range(num_epochs):
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, x in loop:

        data = x.to(device) # Move the batch of data to the selected device

        # Perform a forward pass through the model
        data_reconstructed, mu, logvar = model1(x)  # Get the reconstructed data, mean, and log variance
        
        reconstruction_loss = loss_fn(data_reconstructed, data) # Compute the reconstruction loss
        kl_div = -torch.sum(1+ torch.log(logvar.pow(2)) - mu.pow(2) - logvar.pow(2)) # Compute the KL divergence loss
        
        loss = reconstruction_loss + kl_div # Compute the total loss as the sum of reconstruction loss and KL divergence
        optimizer.zero_grad() # Zero the gradients of the optimizer before backpropagation

        # Perform backpropagation
        loss.backward()
        optimizer.step()  # Update the model parameters using the computed gradients
        
        # Print training loss for this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss / len(data_loader.dataset)}')


# MODEL 2: 
class ConvOnlyVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvOnlyVAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # Output: 32x64x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Output: 64x32x32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: 128x16x16
        self.conv_mu = nn.Conv2d(128, latent_dim, kernel_size=4, stride=1) # Apply convolutional layer to obtain the mean of the latent distribution
        self.conv_logvar = nn.Conv2d(128, latent_dim, kernel_size=4, stride=1) # Apply convolutional layer to obtain the log variance of the latent distribution

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=1)  # Apply the first transposed convolutional layer
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # Apply the second transposed convolutional layer
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # Apply the third transposed convolutional layer
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # Apply the fourth transposed convolutional layer

    def encode(self, x):
        h = F.relu(self.conv1(x)) # Apply the first fully connected layer
        h = F.relu(self.conv2(h)) # Apply the second fully connected layer
        h = F.relu(self.conv3(h)) # Apply the third fully connected layer
        mu = self.conv_mu(h)      # Compute the mean of the latent space distribution
        logvar = self.conv_logvar(h) # Compute the log variance of the latent space distribution
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Compute the standard deviation from the log variance
        eps = torch.randn_like(std)   # Sample epsilon from a standard normal distribution with the same shape as std
        return mu + eps * std         # Generate samples from the latent space

    def decode(self, z):
        h = F.relu(self.deconv1(z))   # Apply the first fully connected layer
        h = F.relu(self.deconv2(h))   # Apply the second fully connected layer
        h = F.relu(self.deconv3(h))   # Apply the third fully connected layer
        return torch.sigmoid(self.deconv4(h)) # Apply the fourth transposed convolution layer and use sigmoid activation to get final output

    def forward(self, x):
        mu, logvar = self.encode(x)   # Encode the input data to obtain the mean and log variance of the latent space distribution
        z = self.reparameterize(mu, logvar) # Reparameterize to obtain latent vectors from the mean and log variance
        return self.decode(z), mu, logvar # Decode the latent vectors to reconstruct the input data
    
# Compute loss (everything here is similar as before)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = ConvOnlyVAE(latent_dim=z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
loss_fn = nn.BCELoss(reduction = "sum")
num_epochs = 10

for epoch in range(num_epochs):
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, x in loop:
        
        data = x.to(device)
        data_reconstructed, mu, logvar = model2(x)
        
        # Compute loss
        reconstruction_loss = loss_fn(data_reconstructed, data)
        kl_div = -torch.sum(1+ torch.log(logvar.pow(2)) - mu.pow(2) - logvar.pow(2))
        
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
        # Print training loss for this epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss / len(data_loader.dataset)}')

# BONUS: Comput images based on model 1
    def generate_images(model, num_images=10, latent_dim=z_dim, device='cpu'):
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Sample from a standard normal distribution
        latent_vectors = torch.randn(num_images, latent_dim).to(device)
        
        # Generate images from latent vectors
        generated_images = model.decode(latent_vectors)
        
        # Post-process images
        generated_images = generated_images.cpu().numpy()
        
        # Visualize images
        fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
        for i in range(num_images):
            axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
            axes[i].axis('off')
        plt.show()

# Example usage
generate_images(model, num_images=10, latent_dim=z_dim, device='cpu')
