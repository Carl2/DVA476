#!/usr/bin/env python
from PIL import Image
import torch
from torchvision import transforms
from pymonad.maybe import Maybe, Just
from common.utils.load_data import (create_image_data_paths, create_image_data_objs,
                             generate_image_transform)
from common.utils import config_helper as ch
from tqdm import tqdm
from pathlib import Path
from functools import reduce

def calculate_mean_std(config:dict, sz:tuple = (128,128)) -> callable:
    temp_transform = transforms.Compose([
        transforms.Resize(sz),
        transforms.ToTensor()
    ])

    def do_calculate_mean_std(paths: list) -> Maybe:

        sum_pixels = torch.zeros(3)
        sum_squared_pixels = torch.zeros(3)

        def process_path(init, path: Path):
            image = Image.open(path).convert('RGB')
            img_tensor = temp_transform(image)  # [3, (sz[0],sz[1])]
            init["sum_pixels"] += img_tensor.sum(dim=[1, 2])  # Sum over H, W
            init["sum_squared_pixels"] += (img_tensor ** 2).sum(dim=[1, 2])
            init["num_pixels"] += sz[0] * sz[1]
            return init


        result = {
            "sum_pixels": sum_pixels,
            "sum_squared_pixels": sum_squared_pixels,
            "num_pixels": 0
        }

        for path in tqdm(paths):
            result = process_path(result,path)


        sum_pixels,num_pixels,sum_squared_pixels = (result['sum_pixels'],
                                                    result['num_pixels'],
                                                    result['sum_squared_pixels'])

        mean = sum_pixels / num_pixels
        std = torch.sqrt(sum_squared_pixels / num_pixels - mean ** 2)

        print(f"Mean {mean} std: {std}")


        config['settings']['mean'] = mean.tolist()
        config['settings']['std'] = std.tolist()
        return Just(config)
    return do_calculate_mean_std


def calculate_dataset_stats(image_data_list: list) -> tuple:
    """
    Memory-efficient version using running statistics.
    """

    temp_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Running sums
    sum_pixels = torch.zeros(3)
    sum_squared_pixels = torch.zeros(3)
    num_pixels = 0

    print("Calculating statistics (memory-efficient)...")
    for img_data in tqdm(image_data_list,desc="Calulating stats...(Normalization)"):
        for i in range(img_data.start_index, img_data.end_index + 1):
            img_path = img_data.path / f"{img_data.file_name}{i}.jpg"

            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                img_tensor = temp_transform(image)  # [3, 64, 64]

                # Update running sums
                sum_pixels += img_tensor.sum(dim=[1, 2])  # Sum over H, W
                sum_squared_pixels += (img_tensor ** 2).sum(dim=[1, 2])
                num_pixels += 64 * 64

            # if i % 500 == 0:
            #     print(f"  Processed {i} images...")

    # Calculate mean and std
    mean = sum_pixels / num_pixels
    std = torch.sqrt(sum_squared_pixels / num_pixels - mean ** 2)

    print("\nCalculated stats:")
    print(f"  Mean: {mean.tolist()}")
    print(f"  Std:  {std.tolist()}")

    return mean.tolist(), std.tolist()



def ensure_normalization_stats(config_path: str) -> callable:

    def do_ensure_normalization_stats(cfg: dict) -> Maybe:
        settings = cfg.get('settings', {})

        if 'mean' in settings and 'std' in settings:
            print("✓ Using cached normalization stats")
            print(f"  Mean: {settings['mean']}")
            print(f"  Std:  {settings['std']}")
            return Just(cfg)

        print("✗ Calculating normalization stats from training data...")

        # Create temp pipeline without normalization,
        # This will calculate the normalization.
        temp_result = Just(cfg) \
            .bind(create_image_data_paths(ch.DatasetType.TRAIN)) \
            .bind(create_image_data_objs) \
            .map(calculate_dataset_stats)

        if not temp_result.is_just():
            return Maybe(f"Failed to calculate stats: {temp_result.value}",
                         monoid=False)

        # Update config
        if 'settings' not in cfg:
            cfg['settings'] = {}
        cfg['settings']['mean'] = temp_result.value[0]
        cfg['settings']['std'] = temp_result.value[1]

        # Save to file, as a cache...
        save_config_to_file(config_path, cfg)
        print(f"✓ Stats saved to {config_path}")

        return Just(cfg)
    return do_ensure_normalization_stats


def save_config_to_file(path: str, config: dict):
    """Save config to JSON file."""
    import json
    clean_config = {k: v for k, v in config.items()
                    if not k.startswith('_')}
    with open(path, 'w') as f:
        json.dump(clean_config, f, indent=2)
