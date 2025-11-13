# Face Generation Autoencoder
Training and evaluating convolutional autoencoders for face reconstruction and generation using the CelebA dataset.

---

## Overview 
This project uses convolutional autoencoders for learning representations of human faces. The goal was to train unsupervised learning models so that the decoder could generate new faces using a latent vector. 
By using different model architectures and training methods, this project demonstrates how autoencoders can be used to compress and reconstruct images as well as generate new images.

For an in-depth explanation of methods and results, see the project report: 
[View Full Report (PDF)](https://github.com/H-Konrad/project_reports/blob/main/face_generation_autoencoder_write_up.pdf)

---

## Repository Structure

The dataset and trained models are not included in this repository. The CelebA dataset is publicly available and can be downloaded online. 

### Code
- `generation.py` : Functions for generating new images. 
- `models.py` : Encoder and autoencoder models. 
- `testing.py` : Functions for testing model accuracy. 
- `training.py` : Functions for training the encoder and autoencoder models. 

### Notebook
- `project.ipynb` : Notebook containing experiments, model training and testing. 
