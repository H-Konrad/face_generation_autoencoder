import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def latent_vector_statistics(model, testing_loader, device):
    """
    A function that gets the mean and standard deviation for latent vectors per value 
    ------
    inputs
    ------
    model: pytorch model, the model that will be used
    testing_loader: data loader, testing data for the model 
    device: hardware, either the CPU or GPU
    -------
    outputs
    -------
    single_mean: tensor, means for each position in the latent vector
    single_std: tensor, stds for each position in the latent vector
    """
    model.eval()
    all_latent_vectors = []
    for i in range(3): # number can be changed to get a more accurate mean and std
        with torch.no_grad():
            for images, labels in testing_loader:
                images = images.to(device)
                latent_vectors = model.encoder(images)
                all_latent_vectors.append(latent_vectors.cpu())

    # mean and std for each batch
    means, stds = [], []
    for latent in all_latent_vectors:
        means.append(latent.mean(dim = 0))
        stds.append(latent.std(dim = 0))

    # means for the mean and std for each tensor
    value_1, value_2 = 0, 0
    for m in means:
        value_1 += m
    tensor_mean = value_1/len(means)

    for s in stds:
        value_2 += s
    tensor_std = value_2/len(stds)

    return tensor_mean, tensor_std

def image_creation(model, device, mean, std, count):
    """
    A function that creates a new random image
    ------
    inputs
    ------
    model: pytorch model, the model that will be used
    device: hardware, either the CPU or GPU
    mean: tensor, the mean of the latent vector
    std: tensor, the std of the latent vector
    count: integer, the number of images that should be created 
    -------
    outputs
    -------
    images: generated images
    """
    # generating random latent vectors
    latent_vectors = [torch.normal(mean = mean, std = std).unsqueeze(0).to(device) for i in range(count)]        
    images = []    
    
    model.eval()
    with torch.no_grad():
        for vectors in latent_vectors:
            new_images = model.decoder(vectors)
            images.append(new_images[0].cpu())
            
    return images

def custom_image_creation(model, device, data):
    """
    A function that creates a custom image
    ------
    inputs
    ------
    model: pytorch model, the model that will be used
    device: hardware, either the CPU or GPU
    data: list, a list of latent vectors
    -------
    outputs
    -------
    images: generated images
    """
    images = []
    model.eval()
    with torch.no_grad():
        for vector in data:
            custom_images = model.decoder(vector.float().unsqueeze(0).to(device))
            images.append(custom_images[0].cpu())
            
    return images