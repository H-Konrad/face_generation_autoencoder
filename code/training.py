import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

def encoder_training(nepoch, model, criterion, optimiser, training_loader, validation_loader, device, name):
    """
    A function that trains an encoder model and returns loss per epoch as a dictionary
    ------
    inputs
    ------
    nepoch: integer value, times the model will go through the dataset 
    model: pytorch model, the model that will be trained 
    criterion: loss function, how loss is measured
    optimiser: weight updater, updates model weights based on loss
    training_loader: data loader, training data for the model 
    validation_loader: data loader, validation data for the model 
    device: hardware, either the CPU or GPU
    name: string, the model name for saving the weights
    """
    for epoch in range(nepoch):
        # training
        model.train()
        training_loss = 0
        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            training_loss += loss.item()

        # checking
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Training loss: {training_loss/len(training_loader):.10f}, Validation loss: {validation_loss/len(validation_loader):.10f}")

    torch.save(model.encoder.state_dict(), f"{name}.pth")

def autoencoder_training(nepoch, model, criterion, optimiser, training_loader, validation_loader, device, name):
    """
    A function that trains a autoencoder model and returns loss per epoch as a dictionary
    ------
    inputs
    ------
    nepoch: integer value, times the model will go through the dataset 
    model: pytorch model, the model that will be trained 
    criterion: loss function, how loss is measured
    optimiser: weight updater, updates model weights based on loss
    training_loader: data loader, training data for the model 
    validation_loader: data loader, validation data for the model 
    device: hardware, either the CPU or GPU
    name: string, the model name for saving the weights
    """        
    for epoch in range(nepoch):
        model.train()
        training_loss = 0
        for images, labels in training_loader:
            images = images.to(device)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimiser.step()
            training_loss += loss.item()

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                validation_loss += loss.item()

        print(f"Epoch: {epoch + 1}, Training loss: {training_loss/len(training_loader):.10f}, Validation loss: {validation_loss/len(validation_loader):.10f}")

    torch.save(model.state_dict(), f"{name}.pth")