#!/usr/bin/env python
import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from pymonad.maybe import Just, Maybe

from img import extract, plot
from nn import classifier, train_plot
from common.utils import coil_utils

from pdb import set_trace
from cnn import cnn_main


def _resolve_path(arg_value):
    """Resolves and expands a Path argument, returns None if input is None."""
    if arg_value:
        return arg_value.expanduser().resolve()
    return None


def model_execution(model, train_tensor_dict: Maybe,
                    validation_tensor_dict: Maybe) -> Maybe:
    """Execute model training with the given training and validation data.

    Args:
        model: The neural network model to train
        train_label_feature: Maybe monad containing training labels and features
        validation_label_feature: Maybe monad containing validation labels and
        features

    Returns:
        Maybe: A Maybe monad containing training history if successful, or an
        error message if validation data is missing
    """
    # Extract normalization stats from training data
    if train_tensor_dict.is_nothing():
        return Maybe(value=f"training data is Nothing {train_tensor_dict.value}"
                     , monoid=False)

    train_data = train_tensor_dict.value['tensor']
    # train_data, norm_stats = train_label_feature.value  # Unpack stats

    # # Apply SAME stats to validation
    # validation_data = validation_label_feature.map(
    #     lambda data: utils.to_tensor_and_normalize(
    #         data,
    #         stats_to_use=norm_stats
    #     )
    # )

    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(model.parameters(), lr=0.001)

    if validation_tensor_dict.is_nothing():
        print("Validation is nil, no training is done.")
        return Maybe(value="No validation data!", monoid=False)
    valid_dict = validation_tensor_dict.value

    maybe_history = train_tensor_dict.bind(
        classifier.train_model(
            model=model,
            optimizer=opti,
            loss=criterion,
            epochs=50,
            validation_label=valid_dict['labels'],
            validation_data=valid_dict['normalized'])) \
                                .bind(classifier.save_model("Ann_model.pt"))



    return maybe_history

def process_image(img: Path, display: bool = False):

    maybe_img = extract.read_image(img) \
                      .bind(coil_utils.extract_label("file")) \
                      .bind(extract.transform_grayscale("raw")) \
                      .bind(extract.extract_statistical_features('gray')) \
                      .bind(extract.color_histogram("raw", "gray", 30)) \
                      .bind(extract.edge_detection("gray", 20, 200)) \
                      .bind(extract.extract_hog_descriptor("gray"))

    if maybe_img.is_nothing():
        print(f"Error: {maybe_img.value}")
        return {}
    feature_vector = extract.create_feature_vector()(maybe_img.value)

    if display:
        plot.display_features(maybe_img.value)
    return {"feature": feature_vector, "label": maybe_img.value["label"]}



def batch_training_data(directory: Path, stats: dict) -> Maybe:
    maybe_tensor_dict = Just(directory).bind(coil_utils.read_files()) \
                                       .bind(coil_utils.process_images(process_image,
                                                                  feature_key="feature",
                                                                  label_key="label")) \
                                       .map(coil_utils.convert_matrix) \
                                       .bind(coil_utils.get_stats(stats)) \
                                       .map(coil_utils.normalize_tensor)

    # The return is a maybe → dict → keys: labels,tensor,normalized, stat{mean,std},
    if maybe_tensor_dict.is_nothing():
        return Maybe(value=f"Failed to process files: {maybe_tensor_dict.value}"
                     , monoid=False)
    return maybe_tensor_dict


def run_test(model_data: Path, test_path: Path ):
    vals = batch_training_data(test_path, {}) \
        .bind(classifier.load_model(model_data)) \
        .map(train_plot.test_plot)

    print(f"WTF {vals}")






def main():
    parser = argparse.ArgumentParser(
    description='Split COIL-100 dataset into train/validation/test sets by rotation angle',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Example:
  coilSplit -source <image directory>


    """)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-source', help='Train data path', type=Path)
    group.add_argument('-img',  help='Single file')
    group.add_argument('-test', help='Test data path', type=Path)
    parser.add_argument('-validation', help='Validation data path', type=Path)
    parser.add_argument('-load', help='Model to load', type=Path)
    parser.add_argument('-cnn', help='Run with CNN instead of ANN', type=Path)
    args = parser.parse_args()



    model = classifier.ClassifierNN()

    if args.img:
        process_image(args.img, True)
    elif args.source and args.validation:

        if args.cnn:
            cnn_config = {}
            cnn_config['config_file'] = args.cnn.expanduser().resolve()
            cnn_config['train_dir'] = args.source.expanduser().resolve()
            cnn_config['validation_dir'] = args.validation.expanduser().resolve()


            print(f"Running CNN mode with config: {args.cnn}")
            cnn_main.train_cnn(cnn_config)
            return


        train_tensor_dict = batch_training_data(args.source, {})
        if train_tensor_dict.is_nothing():
            print(f"could not read batch {train_tensor_dict.value}")

        stats = train_tensor_dict.value['stats']

        validation_tensor_dict = batch_training_data(args.validation, stats)

        print(f"""
        Processed Train: {len(train_tensor_dict.value['labels'])} files
        Validation: {len(validation_tensor_dict.value['labels'])} files""")

        # train model; If successful a plot the history
        model_execution(model, train_tensor_dict, validation_tensor_dict) \
            .map(train_plot.TrainPlot)

    elif args.test and args.load:
        if args.cnn:
            cnn_config = {}
            cnn_config['config_file'] = args.cnn.expanduser().resolve()
            cnn_config['load_model'] = args.load.expanduser().resolve()
            cnn_config['test_dir'] = args.test.expanduser().resolve()
            cnn_main.run_inference(cnn_config)
            return

        run_test(args.load, args.test)

    else:
        print(f"Not working yet")




if __name__ == '__main__':
    main()
