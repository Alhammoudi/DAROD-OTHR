import os
import json
import tensorflow as tf
from darod.models import model
from darod.utils import data_utils, bbox_utils

def save_predictions_to_txt(predictions, filenames, output_dir):
    """
    Save predictions to .txt files in the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for prediction, filename in zip(predictions, filenames):
        # Handle filenames as bytes or strings
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        elif not isinstance(filename, str):
            filename = str(filename)  # Convert to string if it's not already

        # Extract predicted boxes, labels, and scores
        boxes = prediction.get('boxes', [])
        labels = prediction.get('labels', [])
        scores = prediction.get('scores', [])
        
        if not boxes:
            continue  # Skip if no valid predictions
        
        # Normalize coordinates and prepare lines for the .txt file
        lines = []
        for box, label, score in zip(boxes, labels, scores):
            y1, x1, y2, x2 = box  # Assuming box format is [y1, x1, y2, x2]
            lines.append(f"{label} {score:.6f} {y1:.6f} {x1:.6f} {y2:.6f} {x2:.6f}")
        
        # Save to .txt file
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))

def main():
    # Load config.json
    config_path = r"C:\Users\oalha\Desktop\darod\darod-main\logs\darod_othr\config.json"  # Replace with the actual path
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Prepare the batched test dataset
    test_dataset, _ = data_utils.prepare_dataset(split="test", config=config, seed=config["training"]["seed"])

    # Load the trained model
    anchors = bbox_utils.anchors_generation(config, train=False)
    darod_model = model.DAROD(config, anchors)
    darod_model.build(input_shape=(None, config["model"]["input_size"][0], config["model"]["input_size"][1], 1))
    darod_model.load_weights(os.path.join(os.path.dirname(config_path), "best-model.h5"))  # Corrected path

    # Output directory for predictions
    output_dir = config.get('output_dir', './predictions')

    # Run inference and save predictions
    for batch in test_dataset:
        spectrums, filenames = batch[0], batch[-1]
        if spectrums.shape[0] == 0:
            continue  # Skip if no valid frames in the batch
        
        # Ensure filenames are valid strings or bytes
        filenames = [f.decode('utf-8') if isinstance(f, bytes) else str(f) for f in filenames]

        _, _, _, _, _, decoder_output = darod_model(spectrums, training=False)
        pred_boxes, pred_labels, pred_scores = decoder_output

        # Prepare predictions in the required format
        predictions = []
        for boxes, labels, scores in zip(pred_boxes.numpy(), pred_labels.numpy(), pred_scores.numpy()):
            predictions.append({
                'boxes': boxes.tolist(),
                'labels': labels.tolist(),
                'scores': scores.tolist()
            })

        # Save predictions to .txt files
        save_predictions_to_txt(predictions, filenames, output_dir)

if __name__ == "__main__":
    main()