# diffusion-distillation
Implemented DDIM Diffusion and model distillation for fast and deterministic sampling as proposed by the paper: [*Progressive Distillation For Fast Sampling Of Diffusion Models*](https://arxiv.org/abs/2202.00512)

**Generated Images**

<img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/image_0_256.png" width="200" height="200"> <img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/image_1_256.png" width="200" height="200"> <img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/image_2_256.png" width="200" height="200"> <img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/image_3_256.png" width="200" height="200">

## What and how?

### Diffusion models
Diffusion models are latent variable generative models that define a Markov chain of diffusion steps to slowly add random noise to the data, transforming it into an isotropic Gaussian distribution. The model then learns to reverse this diffusion process, allowing it to generate new samples that resemble the original training data.

### Previous method: DDPM
In the previous project I implemented DDPM model, where the model learned to predict noise in a noisy image created by forward diffusion process. Then we started from a pure noisy image and sampled images by iteratively subtracting noise from the initial image. This process had 2 major limitations - dull/not sharp/blurry images and slow sampling speed.

### New method: DDIM + Distillation = Progressive distillation
In this project I aimed to solve these 2 limitations by making 3 changes as presented by the orignal paper:

**1. DDIM Sampler**:

**2. Model Parameterization**:

**3. Distillation**:   

## Comparing results

<img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/ddim-vs-ddpm.png" width="1000" height="250">

**Distillation: 256 sampling steps --> 2 sampling steps**

<img src="https://github.com/Avenger-py/diffusion-distillation/blob/main/assets/compare-distillation.png" width="1000" height="280">

## Dataset
I used Stanford Cars Dataset, which contains about 16k car images. Steps to download dataset: https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616

## Requirements
### Software
- torch
- torchvision
- scipy
- numpy
- matplotlib
- tqdm
- Jupyterlab

### Hardware
I used on-demand cloud GPU service vast.ai and trained the model on RTX 4090 (24gb vram).

## How to run?
1. Clone the repo
2. Install the requirements mentioned above
3. Place the dataset inside `data` folder in the working directory
4. Run `Diffusion_distillation.ipynb`

## Training
Training settings are similar to my previous project: [smol-Diffusion](https://github.com/Avenger-py/smol-Diffusion)
In case of distillation, I decreased the learning rate by 10x

## Resources and Acknowledgments
Based on the papers: 
1. [*Progressive Distillation For Fast Sampling Of Diffusion Models*](https://arxiv.org/abs/2202.00512)
2. [*Denoising Diffusion Implicit Models*](https://arxiv.org/pdf/2010.02502)

Some inspiration taken from: https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch 


