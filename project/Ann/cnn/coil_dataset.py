#!/usr/bin/env python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from PIL import Image
from common.utils.config_helper import DatasetType, get_key_fn
from common.utils import coil_utils
from nn import cnn_classifier
import re
from functools import reduce
from pymonad.maybe import Maybe,Just
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

@dataclass
class CoilImageData:
    file_name: str
    label: int
    transformer: callable



class CoilDataset(Dataset):
    def __init__(self, image_data_list: list[CoilImageData]):
        self.samples=image_data_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_data = self.samples[idx]

        # Load image
        image = Image.open(image_data.file_name).convert('RGB')

        # Apply transforms
        if image_data.transformer:
            image = image_data.transformer(image)

        return image, image_data.label




def create_data_loader(json_content: dict[str,str], dataset_type: DatasetType) -> callable:
    """
    Create a DataLoader factory function with configuration based on dataset type.

    Args:
        config: Configuration dictionary
        dataset_type: Type of dataset (TRAIN, VALIDATE, TEST)

    Returns:
        A function that takes a Dataset and returns Maybe[DataLoader]
    """
    get_key = get_key_fn(json_content, dataset_type)

    def do_create_data_loader(config: dict) -> Maybe:
        dataset = config.get('dataset')
        if dataset is None:
            return Maybe(value=f"dataset not found in loader", monoid=False)
        # Extract all needed config values
        batch_size_maybe = get_key("batch_size")
        shuffle_maybe = get_key("shuffle")
        num_workers_maybe = get_key("num_workers")
        pin_memory_maybe = get_key("pin_memory")

        # Check if all required keys were found
        # Using a more functional approach
        configs = [
            ("batch_size", batch_size_maybe),
            ("shuffle", shuffle_maybe),
            ("num_workers", num_workers_maybe),
            ("pin_memory", pin_memory_maybe)
        ]

        for name, maybe_val in configs:
            if not maybe_val.monoid:
                return Maybe(
                    value=f"Failed to create DataLoader: {maybe_val.value}",
                    monoid=False
                )

        # All values are valid, extract them
        batch_size = batch_size_maybe.value
        shuffle = shuffle_maybe.value
        num_workers = num_workers_maybe.value
        pin_memory = pin_memory_maybe.value

        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

            print(f"[{dataset_type.name}] DataLoader created:")
            print(f"  - Batches (size {batch_size}): {len(dataloader)}")
            print(f"  - Total samples: {len(dataloader) * batch_size}")
            print(f"  - Shuffle: {shuffle}")

            return Just(dataloader)

        except Exception as e:
            return Maybe(
                value=f"Error creating DataLoader: {str(e)}",
                monoid=False
            )

    return do_create_data_loader




def check_coil_class_distribution(dataset):
    """Check class distribution in COIL dataset"""

    # Get all labels from dataset
    all_labels = [image_data.label for image_data in dataset.samples]

    # Count distribution
    distribution = Counter(all_labels)

    print(f"\nCOIL Dataset Distribution:")
    print(f"Total samples: {len(all_labels)}")
    for class_idx in sorted(distribution.keys()):
        count = distribution[class_idx]
        percentage = (count / len(all_labels)) * 100
        print(f"  Object {class_idx + 1}: {count:4d} ({percentage:.1f}%)")

    return distribution



def convert_to_coil_dataset(image_objs_fn: callable) -> callable:
    def do_convert_to_coil_dataset(config: dict) -> Maybe:
        image_objs = image_objs_fn(config)
        dataset = CoilDataset(image_objs)
        config['dataset'] = dataset
        check_coil_class_distribution(dataset)
        return Just(config)
    return do_convert_to_coil_dataset






def create_image_data_objs(file_label_fn:callable, transform_fn: callable) -> callable:
    def do_create_image_data_objs(config:dict) -> Maybe:
        files_labels = file_label_fn(config)
        transformer = transform_fn(config)

        image_objs = []

        for item in files_labels:
            image_objs.append(CoilImageData(item['file'], item['label'], transformer))

        config['image_objs'] = image_objs


        return Just(config)
    return do_create_image_data_objs






def _extract_label(init:list, file_name:str):
    match = re.search(r'obj(\d+)__', str(file_name))
    if match:
        init.append({"file": file_name, "label": int(match.group(1)) -1})
        return init

    print(f"unable to extract label from <{file_name}> (skipping)")

    return init




def image_data_paths(dataset_type: DatasetType, settings: dict) -> callable:
    def do_image_data_paths(files: list) -> Maybe:

        file_label_dict = reduce(_extract_label, files, [])

        settings['file_label'] = file_label_dict
        settings['type'] = dataset_type

        return Just(settings)
    return do_image_data_paths


def create_transform(mean_fn: callable, std_fn: callable, type_fn: callable, shape: tuple) -> callable:
    def do_create_transform(config) -> Maybe:
        mean = mean_fn(config)
        std = std_fn(config)
        dataset_type = type_fn(config)

        transformer = None
        if dataset_type is None:
            return Maybe("Unable to get dataset_type", monoid=False)

        if dataset_type == DatasetType.TRAIN:
            transformer = transforms.Compose([
                transforms.Resize(shape),           # Resize to Shape
                transforms.RandomHorizontalFlip(),     # Data augmentation
                transforms.RandomRotation(10),         # Random rotation ±10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
                transforms.ToTensor(),                 # Convert PIL Image to Tensor
                transforms.Normalize(                  # Normalize to mean=0, std=1
                    mean=mean,
                    std=std
                )
            ])
        else:
            transformer = transforms.Compose([
                transforms.Resize(shape),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=mean ,
                    std=std
                )
            ])

        config["transformer"] = transformer
        return Just(config)
    return do_create_transform




def create_model(model_settings_fn: callable) -> callable:

    m_num_classes = model_settings_fn("num_classes")
    m_dropout_rate = model_settings_fn("dropout_rate")
    m_device = model_settings_fn("device")

    if (m_num_classes.is_nothing()
        or m_dropout_rate.is_nothing()
        or m_device.is_nothing()):
        return lambda config: Maybe(value=f"missing model_settings", monoid=False)


    num_classes = m_num_classes.value
    dropout_rate = m_dropout_rate.value
    device = m_device.value


    def do_create_model(configs: dict) -> Maybe:
        """Create and initialize the CNN model."""
        try:

            model = cnn_classifier.SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
            model = model.to(device)

            # Add model and device to configs
            configs["model"] = model
            configs["device"] = device

            print(f"Model created and moved to {device}")
            return Just(configs)
        except Exception as e:
            return Maybe(value=f"Failed to create model: {str(e)}",monoid=False)


    return do_create_model
