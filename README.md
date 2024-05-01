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
* `generate_from_pretrained_diff.ipynb` is the note book for generating new samples using the pretrained diffusion model

