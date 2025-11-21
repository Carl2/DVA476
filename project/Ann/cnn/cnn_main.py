#!/usr/bin/env python
from pymonad.maybe import Maybe,Just
import json
from cnn import coil_dataset
from common.utils import coil_utils,config_helper,load_data
from common.utils.config_helper import DatasetType
from common.utils import stats
from img import extract
from pathlib import Path
import torch
from nn import cnn_classifier
from common.utils import plotter



def settings_fn(key:str):
    def fn(config:dict):
        return config["settings"][key]
    return fn


def load_config(config: dict)->Maybe:
    config_path = config.get("config_file", None)
    if config_path is None:
        return Maybe(value="Not assigned config", monoid=False)

    p = Path(config_path)
    # if relative, resolve it relative to the script that called this module
    if not p.is_absolute():
        # use the directory of the main script (safer than module's directory)
        p = (Path(__file__).parent.parent / p).resolve()
        # alternatively use Path.cwd().joinpath(p).resolve() if you prefer CWD semantics

    try:
        with open(p, 'r') as f:
            json_config = json.load(f)
            config['config'] = json_config
            #config.update(json_config)
            config['config_file'] = str(p)
        return Just(config)
    except FileNotFoundError:
        return Maybe(f"config not found {p}", monoid=False)
    except json.JSONDecodeError as e:
        return Maybe(f"Invalid JSON in config file {e}", monoid=False)



def process_image(img: Path):
    # process and return.
    # maybe_img = extract.read_image(img) \
    #     coil_utils.extract_label("file")
    # Read image returns a {"raw": ......, "file":....}
    # extract_label takes out the file part,
    maybe_labels = extract.read_image(img) \
                          .bind(coil_utils.extract_label("file"))
    if maybe_labels.is_nothing():
        return None

    return {"feature": maybe_labels.value['raw'], "label": maybe_labels.value["label"]}



def save_settings(config_file: Path) -> callable:
    def do_save_settings(content: dict) -> Maybe:
        try:
            with open(config_file, 'w') as f:
                json.dump(content, f, indent=4)
            print(f"🔥 config saved: {config_file} ✓", sep="", end="")
            return Just(content)
        except IOError as e:
            return Maybe(value=f"Failed to save settings to {config_file}: {e}", monoid=False)
        except Exception as e:
            return Maybe(value=f"Unexpected error saving settings: {e}", monoid=False)
    return do_save_settings





def train_cnn(config: dict)->Maybe:
    #Before continuing i need to figure out if i have
    # Normalized values
    #.bind(stats.ensure_ensure_normalization_stats(config['config_file']))
    # check if config with key "test" , "load_model"


    maybe_config = Just(config).bind(load_config)

    if maybe_config.is_nothing():
        return maybe_config

    config = maybe_config.value
    json_content = config.get("config")
    if json_content is None:
        return Maybe(value=f"Setting not found in configuration", monoid=False)

    mean = json_content.get("settings", None).get("mean",None)
    std = json_content.get("settings", None).get("std",None)

    if std is None or mean is None:

        maybe_mean_std = Just(config['train_dir']) \
            .bind(coil_utils.read_files()) \
            .bind(stats.calculate_mean_std(json_content)) \
            .bind(save_settings(config['config_file']))


    else:
        print(f"Cached Mean and std")

    # 1. read the files... This creates a list of files to be used.
    # 2. For each file we neeed to get the file(path and name), label, type (TRAIN,TEST,...)
    #    2.2 "file_label" key is the "file" , "label" (settings["file_label"]["file"|"label"])
    # p settings['file_label'][0]
    # {'file': PosixPath('/home/calle/git/personal/learn-torch/project/train/obj29__60.png'), 'label': 28}
    #
    #    2.3 "type" , which type train,test or validation.
    # 3. generate a image transform. "transformer"
    # 4. create image data objs. "image_objs"
    # 5. create Dataset
    # 6. create DataLoader.

    model_settings_fn = config_helper.get_key_fn(json_content, config_helper.DatasetType.MODEL_SETTINGS)
    settings_fns = config_helper.get_key_fn(json_content,
                                            config_helper.DatasetType.SETTINGS)


    type_fn = lambda config: config['type']
    shape = (model_settings_fn("image_width").value,
            model_settings_fn("image_height").value)
    mean_fn = settings_fn('mean')
    std_fn = settings_fn('std')


    file_label_fn = lambda config: config['file_label']
    transform_fn = lambda config: config['transformer']
    image_objs_fn = lambda config: config['image_objs']

    maybe_train_dataloader = Just(config['train_dir']) \
        .bind(coil_utils.read_files()) \
        .bind(coil_dataset.image_data_paths(DatasetType.TRAIN,json_content)) \
        .bind(coil_dataset.create_transform(mean_fn,std_fn,type_fn,shape)) \
        .bind(coil_dataset.create_image_data_objs(file_label_fn, transform_fn)) \
        .bind(coil_dataset.convert_to_coil_dataset(image_objs_fn)) \
        .bind(coil_dataset.create_data_loader(json_content, config_helper.DatasetType.TRAIN))

    maybe_validate_dataloader = Just(config['validation_dir']) \
        .bind(coil_utils.read_files()) \
        .bind(coil_dataset.image_data_paths(DatasetType.VALIDATE,json_content)) \
        .bind(coil_dataset.create_transform(mean_fn,std_fn,type_fn,shape)) \
        .bind(coil_dataset.create_image_data_objs(file_label_fn, transform_fn)) \
        .bind(coil_dataset.convert_to_coil_dataset(image_objs_fn)) \
        .bind(coil_dataset.create_data_loader(json_content, config_helper.DatasetType.VALIDATE))
    print(f"Total number samples: {len(maybe_validate_dataloader.value.dataset.samples)} ")

    print(f"""
    training data loaded : {len(maybe_train_dataloader.value)}
    Validation data loaded : {len(maybe_validate_dataloader.value)}
    """)

    config["train_loader"] = maybe_train_dataloader.value
    config["val_loader"] = maybe_validate_dataloader.value

    # Time for creating the training phase.
    # the pipline includes:
    # 1. create the model , this needs the config since the config
    #    holds the training/validate Dataloader.
    # 2. Creates a optimizer using the model_settings.learning_rate
    # 3. Creates loss function Using crossEntropyLoss (gradient descent)
    # 4. Running the training loop.

    model_settings_fn = config_helper.get_key_fn(json_content, config_helper.DatasetType.MODEL_SETTINGS)

    get_settings_fn = config_helper.get_key_fn(json_content, config_helper.DatasetType.SETTINGS)
    logging_fn = config_helper.get_key_fn(json_content, config_helper.DatasetType.LOGGING)


    csv_file  = logging_fn("metrics_csv").value
    result = Just(config) \
        .bind(cnn_classifier.create_cnn_model(model_settings_fn)) \
        .bind(config_helper.create_optimizer(model_settings_fn)) \
        .bind(config_helper.create_loss_function) \
        .bind(config_helper.training_loop(model_settings_fn,
                                          get_settings_fn,
                                          logging_fn)) \
        .bind(plotter.plot_training_history(csv_file))


    return result

def run_inference(config: dict):
    maybe_config = Just(config).bind(load_config)



    if maybe_config.is_nothing():
        return maybe_config

    config = maybe_config.value
    json_content = config.get("config")
    if json_content is None:
        return Maybe(value=f"Setting not found in configuration", monoid=False)

    model_settings_fn = config_helper.get_key_fn(json_content, config_helper.DatasetType.MODEL_SETTINGS)

    mean_fn = settings_fn('mean')
    std_fn = settings_fn('std')
    type_fn = lambda config: config['type']
    shape = (model_settings_fn("image_width").value,
             model_settings_fn("image_height").value)
    file_label_fn = lambda config: config['file_label']
    transform_fn = lambda config: config['transformer']
    image_objs_fn = lambda config: config['image_objs']
    dataset_type = DatasetType.TEST
    # 1. Create the model. for this:
    # 2. Load the model, The model can be found config["load_model"]
    maybe_test_dataloader = Just(config['test_dir']) \
        .bind(coil_utils.read_files()) \
        .bind(coil_dataset.image_data_paths(dataset_type,json_content)) \
        .bind(coil_dataset.create_transform(mean_fn,std_fn,type_fn,shape)) \
        .bind(coil_dataset.create_image_data_objs(file_label_fn, transform_fn)) \
        .bind(coil_dataset.convert_to_coil_dataset(image_objs_fn)) \
        .bind(coil_dataset.create_data_loader(json_content, dataset_type))

    if maybe_test_dataloader.is_nothing():
        return maybe_test_dataloader

    config["dataloader"] = maybe_test_dataloader.value


    m_config = maybe_config \
        .bind(coil_dataset.create_model(model_settings_fn)) \
        .bind(config_helper.load_model) \
        .bind(config_helper.create_optimizer(model_settings_fn)) \
        .bind(config_helper.create_loss_function) \
        .bind(run_test_loop(model_settings_fn)) \
        .bind(plotter.plot)


    return m_config


def run_test_loop(settings_fn:callable) -> callable:
    num_classes = settings_fn("num_classes").value

    def do_run_test_loop(config:dict) -> Maybe:
        config["num_classes"] = num_classes
        return config_helper.inference_evaluation(config)
    return do_run_test_loop
