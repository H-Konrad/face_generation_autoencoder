import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def encoder_testing(model, testing_loader, device):
    """
    A function that tests the models accuracy
    ------
    inputs
    ------
    model: pytorch model, the model that will be tested
    testing_loader: data loader, testing data for the model 
    device: hardware, either the CPU or GPU
    """
    correct_all_labels, correct_label, total = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in testing_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            prediction = (outputs > 0.5).int() # changes model outputs to predictions 
            total += labels.size(0) # adds the batch number
            correct_label += (prediction == labels).sum().item() # checks if the label is correct 

    print(f"label accuracy: {(correct_label/(total * 40)) * 100:.2f}%")

def image_reconstruction(model, testing_loader, device, count):
    """
    A function that returns reconstructed images with their original in two lists
    ------
    inputs
    ------
    model: pytorch model, the model that will be tested
    testing_loader: data loader, testing data for the model 
    device: hardware, either the CPU or GPU
    count: integer, amount of reconstructed images to be picked 
    -------
    outputs
    -------
    original_images: list, input images
    reconstructed_images: list, output images
    """
    model.eval() 
    with torch.no_grad():
        images, labels = next(iter(testing_loader)) # gets only the first batch 
        images = images.to(device)
        batch_size = images.size(0) # size to get the largest number for random integer
        outputs = model(images)

    original_images, reconstructed_images = [], []
    for i in range(count):
        # to get identical images across models, set 'shuffle = False' in the dataloader and comment out the line below
        #i = random.randint(0, batch_size)
        original_images.append(images[i].cpu())
        reconstructed_images.append(outputs[i].cpu())
        
    return original_images, reconstructed_images