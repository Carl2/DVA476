#!/usr/bin/env python
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from pymonad.maybe import Just, Maybe, Nothing
from functools import reduce
import imageio.v3 as iw





def extract_label(file_key:str) -> callable:
    def do_extract_label(image_feature) -> Maybe:
        file_name = image_feature.get(file_key, None)
        if file_name is not None:
            match = re.search(r'obj(\d+)__', str(file_name))
            if match:
                image_feature["label"] = int(match.group(1)) -1
                return Just(image_feature)
            return Maybe(value=f"filename {file_name} did not match the proper label pattern",
                         monoid=False)
        return Maybe(value=f"file key <{file_key}> did not exist", monoid=False)
    return do_extract_label


# def extract_label(file_name:Path) -> Maybe:
#     if file_name is not None:
#         match = re.search(r'obj(\d+)__', str(file_name))
#         if match:
#             return Just(match.group(1))
#     return Maybe(value=f"")



def read_files(*,file_ext:str = "png") -> callable:
    def do_read_files(source_directory: Path) -> Maybe:
        try:
            path = Path(source_directory)
            if not path.exists() or not path.is_dir():
                return Maybe(value=f"Path {source_directory} does not exists ")

            png_files = list(path.glob(f"*.{file_ext}"))

            if not png_files:
                return Maybe(value=f"No existsing file with extension {file_ext}")

            return Just(png_files)
        except Exception as e:
            return Maybe(value=f"Error reading train data: {e}", monoid=False)
    return do_read_files


def process_images(fn: callable,*,feature_key:str, label_key:str) -> callable:
    def do_process_images(list_of_files) -> Maybe:
        if len(list_of_files) == 0:
            return Maybe(value=f"No file in list?", monoid=False)
        with Pool(processes=cpu_count()) as pool:
            feature_vectors = pool.map(fn, list_of_files)

        features, labels = zip(*((item["feature"], item["label"]) for item in feature_vectors))
        # Maybe filter out None from feature vector?.
        # This is a list of feature vectors.
        return Just((labels,features))
    return do_process_images

def convert_matrix(label_feature_tuple: tuple):
    label_list, feature_list = label_feature_tuple
    return {"labels": label_list,
            "data": np.array(feature_list)}


def get_stats(use_stats:dict) -> callable:
    """
    Returns a function that computes or retrieves statistical metrics for tensor data.

    Args:
        use_stats (dict): Dictionary containing pre-computed statistics ('mean' and 'std').
                         If statistics are not provided, they will be computed from the data.

    Returns:
        callable: A function that takes label_data dict and returns a Maybe monad.
                 The returned function computes mean and std statistics for the tensor
                 and stores them in tensor_data dict along with the tensor itself.
    """

    def do_get_stats(label_data: dict) -> Maybe:
        tensor_data = {}
        stats = {}
        tensor = torch.tensor(label_data["data"], dtype=torch.float32)
        stats["mean"] = use_stats.get("mean", tensor.mean(dim=0))
        stats["std"] = use_stats.get("std", tensor.std(dim=0))
        tensor_data['stats'] = stats
        tensor_data["tensor"] = tensor
        tensor_data["labels"] = label_data.get("labels", None)
        return Just(tensor_data)
    return do_get_stats




# def to_tensor_and_normalize(label_feature_tuple: tuple):
#     """Normalize the np array tensor"""
#     label_list, feature_matrix = label_feature_tuple
#     X_train = torch.tensor(feature_matrix, dtype=torch.float32)  # Shape: (51, 8261)
#     mean = X_train.mean(dim=0)  # Shape: (8261,) - one mean per feature (along the columns)
#     std = X_train.std(dim=0)    # Shape: (8261,) - one std per feature
#     return (label_list, ((X_train - mean) / std))



def normalize_tensor(tensor_data: dict) -> Maybe:
    """
    Args:
        stats_to_use: Dict with 'mean' and 'std' to use for normalization
        save_stats: If True, return stats instead of data
    """
    feature_tensor = tensor_data['tensor']
    mean = tensor_data['stats']['mean']
    std = tensor_data['stats']['std']

    tensor_data['normalized'] = (feature_tensor - mean) / std


    return tensor_data


def merge_dicts(one: dict, two: dict):
    return one.update(two)
