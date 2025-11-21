#!/usr/bin/env python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List
from common.utils.load_data import ImageData
import torch
from pymonad.maybe import Maybe,Just
from collections import Counter

class EuroSatDataset(Dataset):
    def __init__(self, image_data_list: List[ImageData], transform=None):
        """
        Args:
            image_data_list: List of ImageData objects
            transform: Optional transform to be applied on images
        """
        self.samples = []  # List of (path, class_idx) tuples
        self.transform = transform

        # Create class_name -> index mapping (labels)
        self.class_to_idx = {
            img_data.class_name: idx
            for idx, img_data in enumerate(image_data_list)
        }

        # Generate all image paths with their class indices
        # This will add a [path, idx] where path is the image file and idx is the label index.
        for img_data in image_data_list:
            class_idx = self.class_to_idx[img_data.class_name]

            for i in range(img_data.start_index, img_data.end_index + 1):
                img_path = img_data.path / f"{img_data.file_name}{i}.jpg"
                self.samples.append((img_path, class_idx))

    def __len__(self):
        """Returns total number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns one sample at a time"""
        img_path, class_idx = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        return image, class_idx



def check_class_distribution(dataset):
    """Check class distribution in dataset"""


    # Get all labels from dataset
    all_labels = [label for _, label in dataset.samples]

    # Count distribution
    distribution = Counter(all_labels)

    print(f"\nDistribution:")
    print(f"Total samples: {len(all_labels)}")
    for class_idx in sorted(distribution.keys()):
        count = distribution[class_idx]
        percentage = (count / len(all_labels)) * 100
        print(f"  Class {class_idx}: {count:4d} ({percentage:.1f}%)")



def convert_to_dataset(image_data_objs: list):
    """
    Convert a list of image data objects into an EuroSatDataset.

    Args:
        image_data_objs: List of image data objects to be converted into a dataset.

    Returns:
        EuroSatDataset: A dataset containing all the image data objects.

    Note:
        The transformer is the same for all datasets of the same dataset type,
        so it is retrieved from the first image data object.
    """
    dataset = EuroSatDataset(image_data_objs, image_data_objs[0].transformer)
    check_class_distribution(dataset)
    return dataset


def create_data_loader(config: dict) -> callable:
    def do_create_data_loader(dataset: Dataset) -> Maybe:
        batch_size = config.get("batch_size", 32)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=config.get("shuffle", True),
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True)
        )
        print(f"Number of batches(size {batch_size}): {len(dataloader)} ≅ {len(dataloader) * batch_size} Images")

        return Just(dataloader)
    return do_create_data_loader
