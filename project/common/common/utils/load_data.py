#!/usr/bin/env python
from pathlib import Path
from pymonad.maybe import Maybe,Just
from common.utils import config_helper as ch
import re
import itertools
from pdb import set_trace
from dataclasses import dataclass
from torchvision import transforms

@dataclass(kw_only=True)
class BasePath:
    path: str = "/home/calle/git/personal/learn-torch/project/CNN/euro/2750"
    transformer: callable

@dataclass
class ImageData(BasePath):
    start_index: int
    end_index: int
    file_name: str
    class_name: str


def _create_image_data(path: Path) -> callable:

    def do__create_image_data(image_type: str):
        class_path = path.joinpath(image_type)
        return class_path
    return do__create_image_data


def create_image_data_paths(dataset_type: ch.DatasetType) -> callable:
    def do_create_image_data_paths(config: dict) -> Maybe:
        settings_config = ch.get_settings(config, ch.DatasetType.SETTINGS)
        train_config_fn = ch.get_settings(config, dataset_type)

        classes = ch.get_classes(config, dataset_type)

        setting = config.get("settings",{})
        path = setting.get("path", None)
        if path is None:
            return Maybe(value=f"The config file settings did not provide path",
                         monoid=False)

        mean = setting.get("mean", None)
        std = setting.get("std", None)

        class_and_paths = {my_path: Path(path).joinpath(my_path) for my_path in classes}
        return Just({"class_path": class_and_paths,
                     "type": dataset_type,
                     "type_config_fn": train_config_fn,
                     "mean": mean,
                     "std": std
                     })

        return Maybe(value=f"unable to retrive path from config", monoid=False)


    return do_create_image_data_paths


def generate_image_transform(configs: dict):
    dataset_type = configs.get("type", None)
    mean = configs.get("mean",None)
    std = configs.get("std",None)
    # TODO: Get the mean and std.

    transformer = None
    if dataset_type is None:
        return Maybe("Unable to get dataset_type", monoid=False)

    if dataset_type == ch.DatasetType.TRAIN:
        transformer = transforms.Compose([
            transforms.Resize((64, 64)),           # Resize to 64x64
            transforms.RandomHorizontalFlip(),     # Data augmentation
            transforms.RandomRotation(10),         # Random rotation ±10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variation
            transforms.ToTensor(),                 # Convert PIL Image to Tensor
            transforms.Normalize(                  # Normalize to mean=0, std=1
                mean=mean,       #  [0.485, 0.456, 0.406]
                std=std          # [0.229, 0.224, 0.225]
            )
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean , #[0.485, 0.456, 0.406]
                std=std #[0.229, 0.224, 0.225]
            )
        ])

    configs["transformer"] = transformer

    return Just(configs)






def create_image_data_objs(paths_config: dict):
    class_path_dict = paths_config['class_path']
    dataset_type = paths_config['type']
    dataset_config_fn = paths_config['type_config_fn']
    transformer = paths_config.get('transformer', None)

    classes = list(class_path_dict.keys())

    start_end = {class_type:
                 dataset_config_fn(class_type) \
                     .map(ch.get_start_end)
                     for class_type in classes
                 }

    def _convert_objs(sat_type: str, start: int, end: int, path: Path, transformer: callable):
        return ImageData(start, end, f"{sat_type}_", sat_type, path=path, transformer=transformer)

    objs = [_convert_objs(sat_type,indexes.value[0],indexes.value[1],class_path_dict[sat_type], transformer)
            for sat_type,indexes in start_end.items() if indexes.is_just()]


    return Just(objs)





def _matcher(regular_expr, start:int, end: int) -> callable:
    def do__matcher(string_item) -> bool:
        print(f"{string_item.name} start: {start} {end}")

        match = regular_expr.match(str(string_item.name))
        if match:

            index = int(match.group(1))
            #print(f"Got {index}")
            if start <= index <= end:
                return True
        return False

    return do__matcher


def read_data(*,start_index:int, end_index:int, prefix: str) -> callable:
    pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.jpg$')

    def do_read_data(data_path:Path) -> Maybe:
        # read the complete directory
        list_of_all_files = data_path.glob('*.jpg')
        match_fn = _matcher(pattern, start=start_index,end=end_index)

        list_matches = list(filter(match_fn, list_of_all_files))
        return Just(list_matches)

    return do_read_data

#def read_data(train_data_path:Path)->Maybe(list[str]):


def main():


    train_fn = read_data(prefix="AnnualCrop", start_index=1,end_index=5)
    maybe_my_list=train_fn(Path("/home/calle/git/personal/learn-torch/project/CNN/euro/2750/AnnualCrop"))

    if maybe_my_list.is_just():
        print(f"{maybe_my_list.value}")



if __name__ == '__main__':
    main()
