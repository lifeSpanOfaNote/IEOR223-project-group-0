# IEOR223-project-group-0

This is the github repository of the IEOR223 2024 Spring course project of group 0.

## Requirement

We require the following pakcages (open the following links for installation instructions):

* Pytorch https://pytorch.org/get-started/locally/
* Diffusers https://huggingface.co/docs/diffusers/en/installation

Additionally, importing models and datasets from the Huggingface Diffusers Hub sometimes requires login to your Hugging Face account. Please login from the terminal using the following command,
```shell
huggingface-cli login
```
and enter a token. (see https://huggingface.co/docs/hub/security-tokens)

## Diffusion models

The folder `Diffusion_models` contains the following:

* `image_diff.ipynb` is the notebook for creating and training a diffusion model on the ["YiwengXie/102_flowers_small"](https://huggingface.co/datasets/YiwengXie/102_flowers_small) dataset, starting from scratch.
* The `diffusion_for_images_pretrained` folder contains sample images and model weights saved from our previous training processes.
* `generate_from_pretrained_diff.ipynb` is the note book for generating new samples using the pretrained diffusion model saved in `diffusion_for_images_pretrained`, through an [automized pipeline](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline).


## VAE 

The folder `VAE` contains the following:

* `VAE_with_IS_FID.ipynb` is a notebook for creating, training and evaluating a VAE model on the CIFAR-10 dataset from scratch.

## GAN

The folder `GAN` contains the following:

* `DCGAN.ipynb` is a notebook for creating, training and evaluating a DCGAN model on the ["YiwengXie/102_flowers_small"](https://huggingface.co/datasets/YiwengXie/102_flowers_small) dataset from scratch.
* cGAN.py: Training process

model_edge2hats.py: the network structures

ui_edge2hat.py: ui design of .exe and .app

Data folder: hat data for training

models folder: the entire model

Result folder: evolution of images and comparison between edge, ground truth and generated images during the training.

Due to file uploading restriction, the data and model can be downloaded from https://drive.google.com/file/d/1l4o-G7tUo4jmzIWLETKYNHmLQV5PMfBG/view?usp=drive_link


