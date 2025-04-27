import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from darod.models import model
from darod.utils import io_utils, data_utils, bbox_utils, viz_utils

def main():
    args = io_utils.handle_args_viz()

    config_pth = os.path.join(args.path, "config.json")
    with open(config_pth, 'r') as file:
        config = json.load(file)

    dataset = config["data"]["dataset"]
    labels = config["data"]["labels"]
    labels = ["bg"] + labels
    seed = config["training"]["seed"]
    layout = config["model"]["layout"]
    target_id = 0  # args.seq_id
    eval_best = args.eval_best if args.eval_best is not None else False

    # Prepare dataset
    if dataset == "carrada":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    elif dataset == "othr":
        batched_test_dataset, dataset_info = data_utils.prepare_dataset(split="test", config=config, seed=seed)
    else:
        raise NotImplementedError("This dataset doesn't exist.")

    anchors = bbox_utils.anchors_generation(config, train=False)
    faster_rcnn_model = model.DAROD(config, anchors)
    if layout == "2D":
        faster_rcnn_model.build(input_shape=(None, config["model"]["input_size"][0], config["model"]["input_size"][1], 1))
    else:
        fake_input = tf.zeros(shape=(config["training"]["batch_size"], config["model"]["sequence_len"], config["model"]["input_size"][0], config["model"]["input_size"][1], 1))
        _ = faster_rcnn_model(fake_input)
    faster_rcnn_model.summary()

    def restore_ckpt(config, log_pth):
        optimizer = config["training"]["optimizer"]
        lr = config["training"]["lr"]
        momentum = config["training"]["momentum"]
        if optimizer == "SGD":
            optimizer = tf.optimizers.SGD(learning_rate=lr, momentum=momentum)
        elif optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        elif optimizer == "adad":
            optimizer = tf.optimizers.Adadelta(learning_rate=1.0)
        elif optimizer == "adag":
            optimizer = tf.optimizers.Adagrad(learning_rate=lr)
        else:
            raise NotImplemented("Not supported optimizer {}".format(optimizer))
        global_step = tf.Variable(1, trainable=False, dtype=tf.int64)
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=faster_rcnn_model, step=global_step)
        manager = tf.train.CheckpointManager(ckpt, log_pth, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)

    # Restore checkpoint or load best model
    if eval_best:
        faster_rcnn_model.load_weights(os.path.join(args.path, "best-model.h5"))
    else:
        restore_ckpt(config, log_pth=args.path)

    idx = 0
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    for data in batched_test_dataset:
        spectrums, gt_boxes, gt_labels, is_same_seq, seq_id, _, images = data
        if seq_id.numpy()[0] != target_id and dataset == "carrada":
            continue
        else:
            valid_idxs = tf.where(is_same_seq == 1)
            spectrums, gt_boxes, gt_labels = tf.gather_nd(spectrums, valid_idxs), \
                tf.gather_nd(gt_boxes, valid_idxs), tf.gather_nd(gt_labels, valid_idxs)
            if spectrums.shape[0] != 0:
                _, _, _, _, _, decoder_output = faster_rcnn_model(spectrums, training=False)
                pred_boxes, pred_labels, pred_scores = decoder_output

                # Step 1: Save the image without bounding boxes
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(images[0].numpy().astype("uint8"), aspect='auto')
                plt.axis('off')
                image_path = os.path.join(output_dir, f"image_{idx}.png")
                plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print(f"Saved image without bounding boxes: {image_path}")

                # Step 2: Overlay bounding boxes on the saved image
                img = Image.open(image_path)
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(img)

                # RD map details
                rd_map_top_left_x, rd_map_top_left_y = 85, 56  # RD map's top-left corner
                rd_map_width, rd_map_height = 512, 630        # RD map's width and height

                # Overlay bounding boxes
                for box, score in zip(pred_boxes[0], pred_scores[0]):
                    y1, x1, y2, x2 = box.numpy()
                    
                    # Scale bounding box coordinates to RD map dimensions
                    y1 = y1 * rd_map_height + rd_map_top_left_y
                    x1 = x1 * rd_map_width + rd_map_top_left_x
                    y2 = y2 * rd_map_height + rd_map_top_left_y
                    x2 = x2 * rd_map_width + rd_map_top_left_x
                    
                    if score.numpy() > 0.5:  # Filter by confidence score
                        print(f"Adjusted box: [{y1:.2f}, {x1:.2f}, {y2:.2f}, {x2:.2f}], Score: {score.numpy():.2f}")
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(x1, y1 - 5, f"{score.numpy():.2f}", color='red', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

                overlay_path = os.path.join(output_dir, f"overlay_{idx}.png")
                plt.axis('off')
                plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                print(f"Saved overlay image: {overlay_path}")

                idx += 1
                if idx >= 10:  # Limit to 10 predictions
                    break

if __name__ == "__main__":
    main()