import tensorflow as tf
import os

# Enable GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Other imports
from darod.models import model
from darod.trainers.trainer import Trainer
from darod.utils import data_utils, bbox_utils, io_utils

# print path to utils 
print("Path to utils: ", os.path.dirname(os.path.abspath(__file__)))

seed = 42

# Load config from arguments
args = io_utils.handle_args()
config = io_utils.args2config(args)
epochs = config["training"]['epochs']
batch_size = config["training"]["batch_size"]

# Prepare dataset

if config["data"]["dataset"] == "carrada":
    batched_train_dataset, dataset_info = data_utils.prepare_dataset(split="train", config=config, seed=seed)
    num_train_example = data_utils.get_total_item_size(dataset_info, "train")
    #
    batched_test_dataset, _ = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    batched_val_dataset, _ = data_utils.prepare_dataset(split="val", config=config, seed=seed)
else:
    batched_train_dataset, dataset_info = data_utils.prepare_dataset(split="train[:90%]", config=config, seed=seed)
    num_train_example = data_utils.get_total_item_size(dataset_info, "train[:90%]")
    #
    batched_val_dataset, _ = data_utils.prepare_dataset(split="train[90%:]", config=config, seed=seed)
    batched_test_dataset, _ = data_utils.prepare_dataset(split="test", config=config, seed=seed)

labels = data_utils.get_labels(dataset_info)
config["data"]["total_labels"] = len(labels) + 1
labels = ["bg"] + labels
config["training"]["num_steps_epoch"] = num_train_example
config["training"]["seed"] = seed

# Generate anchors
anchors = bbox_utils.anchors_generation(config, train=True)
# Load model
faster_rcnn_model = model.DAROD(config, anchors)

tf.debugging.set_log_device_placement(True)

# Build the model 
_ = faster_rcnn_model(tf.zeros([1, config["model"]["input_size"][0], config["model"]["input_size"][1], config["model"]["input_size"][-1]]))
faster_rcnn_model.summary()

# Train model 
trainer = Trainer(config=config, model=faster_rcnn_model, experiment_name=config["log"]["exp"],
                  backbone=config["model"]["backbone"], labels=labels, backup_dir=args.backup_dir)
trainer.train(anchors, batched_train_dataset, batched_val_dataset)
