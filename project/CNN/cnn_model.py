#!/usr/bin/env python

import argparse
import json
# from utils.load_data import (read_data,create_image_data_paths,
#                              create_image_data_objs, generate_image_transform)
from common.utils.load_data import (read_data, create_image_data_paths,
                                    create_image_data_objs, generate_image_transform)


from common.utils import config_helper as ch
from common.utils import stats
from nn import dataset,network
from common.utils import plotter
from pymonad.maybe import Maybe, Just
from torchvision import transforms

def run_inference(model_path: str) -> callable:
    def do_run_inference(config: dict) -> Maybe:
        test_settings = config.get("test_settings",None)
        config["model_path"] = model_path
        if test_settings is None:
            return Maybe(f"Config did not provide \"test_settings\" ", monoid=False)

        test_dataloader = Just(config) \
            .bind(create_dataset_pipeline(ch.DatasetType.TEST)) \
            .bind(dataset.create_data_loader(test_settings))

        if test_dataloader.is_nothing():
            return test_dataloader

        config['dataloader'] = test_dataloader.value
        print(f"Run inference: {model_path}")

        result = Just(config) \
            .bind(network.create_model) \
            .bind(network.load_model) \
            .bind(network.create_optimizer) \
            .bind(network.create_loss_function) \
            .bind(network.inference_loop) \
            .bind(plotter.plot)

        if result.is_nothing():
            print(f"ERROR {result.value}")

        return Just(config)
    return do_run_inference


def create_dataset_pipeline(dataset_type: ch.DatasetType) -> callable:
    def do_create_dataset_pipeline(config_objs) -> Maybe:
        return Just(config_objs) \
            .bind(create_image_data_paths(dataset_type)) \
            .bind(generate_image_transform) \
            .bind(create_image_data_objs) \
            .map(dataset.convert_to_dataset)

    return do_create_dataset_pipeline

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CNN Training Script')

    parser.add_argument('--config',
                       type=str,
                       required=True,
                       help='Path to the config JSON file')
    parser.add_argument('--test-model',
                        type=str,
                        help='Path to a pre-trained model .pt file for testing or inference')
    return parser.parse_args()

def load_config(config_path)->Maybe:
    """Load and parse the JSON config file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return Just(config)
    except FileNotFoundError:
        return Maybe(f"config not found {config_path}",monoid=False)
    except json.JSONDecodeError as e:
        return Maybe(f"Invalid JSON in config file {e}",monoid=False)

def main() -> None:
    # Parse command line arguments
    args = parse_args()

    # Load config
    config = load_config(args.config) \
        .bind(stats.ensure_normalization_stats(args.config))

    if args.test_model:
        config.bind(run_inference(args.test_model))
        return

    # TODO: This is a bit of a hack...Need to check if config is
    # acually present.
    if config.is_nothing():
        print(f"error: {config.value}")
        return

    train_settings = config.value["train_settings"]
    validation_settings = config.value["validation_settings"]

    train_dataloader = config \
        .bind(create_dataset_pipeline(ch.DatasetType.TRAIN)) \
        .bind(dataset.create_data_loader(train_settings))

    val_dataloader = config \
        .bind(create_dataset_pipeline(ch.DatasetType.VALIDATE)) \
        .bind(dataset.create_data_loader(validation_settings))

    if train_dataloader.is_nothing():
        print(f"Error: {train_dataloader.value}")
        return

    if val_dataloader.is_nothing():
        print(f"Error: {val_dataloader.value}")
        return



    print(f"""
    training data loaded : {len(train_dataloader.value)}
    Validation data loaded : {len(val_dataloader.value)}
    """)

    config.value["train_loader"] = train_dataloader.value
    config.value["val_loader"] = val_dataloader.value

    logging_csv = config.value["logging"]["metrics_csv"]

    result = config \
        .bind(network.create_model) \
        .bind(network.create_optimizer) \
        .bind(network.create_loss_function) \
        .bind(network.training_loop) \
        .bind(plotter.plot_training_history(logging_csv))

    if result.is_nothing():
        print(f"Training failed: {result.value}")
    else:
        print("Training completed successfully!")
if __name__ == '__main__':
    main()
