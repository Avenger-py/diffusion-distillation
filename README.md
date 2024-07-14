# diffusion-distillation
Implemented DDIM Diffusion and model distillation for fast and deterministic sampling as proposed by the paper: [*Progressive Distillation For Fast Sampling Of Diffusion Models*](https://arxiv.org/abs/2202.00512)

Coming soon ..

## What and how?

### Diffusion models
Diffusion models are latent variable generative models that define a Markov chain of diffusion steps to slowly add random noise to the data, transforming it into an isotropic Gaussian distribution. The model then learns to reverse this diffusion process, allowing it to generate new samples that resemble the original training data.

### Previous method - DDPM
In the previous project I implemented DDPM model, where the model learned to predict noise in a noisy image created by forward diffusion process. Then we started from a pure noisy image and sampled images by iteratively subtracting noise from the initial image. This process had 2 major limitations - dull/not sharp/blurry images and slow sampling speed.

### New method - DDIM + Distillation = Progressive distillation
In this project I aimed to solve these 2 limitations by making 3 changes as presented by the orignal paper:
**1. DDIM Sampler**:

**2. Model Parameterization**:

**3. Distillation**:   

## Comparing results

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
I used on-demand cloud GPU service vast.ai and trained the model for x hours on RTX 4090.

## How to run?
1. Clone the repo
2. Install the requirements mentioned above
3. Place the dataset inside `data` folder in the working directory
4. 

## Training

## Sampling
Sampling is quite slow. I used 15000 timesteps to obtain respectable results.
Sampling is major limitation of DDPM diffusion. The images are blurry, less sharp, less colorful and look like mean of the data.

## Resources and Acknowledgments


