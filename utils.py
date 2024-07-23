import matplotlib.pyplot as plt
import numpy as np
import scipy
from torchvision import transforms
import torchvision
import torch
from PIL import Image


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])


def load_transformed_dataset(img_size, data_root_dir):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=data_root_dir, transform=data_transform, split='train')
    test = torchvision.datasets.StanfordCars(root=data_root_dir, transform=data_transform, split='test')

    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))


def show_any_images(images, titles=None, cols=2):
    num_samples = len(images)
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(images):
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        show_tensor_image(img.detach().cpu())
        if titles:
            plt.title(titles[i])


def save_tensor_image(image, path, size):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    final_image = reverse_transforms(image)

    if size:
        final_image = final_image.resize(size, Image.BICUBIC)

    final_image.save(path)
    print(f"Image saved as {path}")

def equally_spaced_items(lst, N):
    if N <= 0:
        return []
    if N == 1:
        return [lst[-1]]
    
    step = (len(lst) - 1) / (N - 1)
    indices = [int(i * step) for i in range(N - 1)]
    indices.append(len(lst) - 1)
    
    return [lst[i] for i in indices]