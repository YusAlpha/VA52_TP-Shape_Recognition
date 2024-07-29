from torch.utils.data import DataLoader
from torchvision import datasets , transforms
import torch
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize([30, 30]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
}
# Chargement des donnees
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "dataset")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir , x), data_transforms[x]) for x in ["train", "test"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=0) for x in ["train", "test"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}
class_names = image_datasets["train"].classes
