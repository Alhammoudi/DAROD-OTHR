"""raddet dataset."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from . import loader, helper

from scipy import ndimage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# TODO(raddet): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(raddet): BibTeX citation
_CITATION = """
"""


class Othr(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for raddet dataset."""
    VERSION = tfds.core.Version('1.1.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # No need to keep 'spectrum' and 'image' field. Serialize their filenames is
                # enough to load it and allows to save disk space.
        
                'spectrum': tfds.features.Tensor(shape=(535, 577), dtype=tf.float32),
                'image': tfds.features.Image(shape=(None, None, 3)),
                'spectrum/filename': tfds.features.Text(),
                'spectrum/id': tf.int64,
                'sequence/id': tf.int64,
                'objects': tfds.features.Sequence({
                    'area': tf.int64,
                    'bbox': tfds.features.BBoxFeature(),
                    'id': tf.int64,
                    'label': tfds.features.ClassLabel(names=['bragg_lines', 'horizontal_lines', \
                                                             'ionosphere_interference', 'vertical_lines'])
                    # Keep 0 class for BG
                }),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('spectrum', 'objects'),  # Set to `None` to disable
            homepage="",
            citation=_CITATION,
            disable_shuffling=True,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO: Downloads the data and defines the splits
        train_path = "C:\\Users\\oalha\\Desktop\\darod-othr\\datasets\\othr_builder\\data\\train\\"
        test_path = "C:\\Users\\oalha\\Desktop\\darod-othr\\datasets\\othr_builder\\data\\test\\" 
        # TODO Returns the Dict[split names, Iterator[Key, Example]
        return {
            'train': self._generate_examples(train_path, 'train'),
            'test': self._generate_examples(test_path, 'test'),
        }

    def _generate_examples(self, path, input_type="RD"):
        classes_list = ['bragg_lines', 'horizontal_lines', 'ionosphere_interference', 'vertical_lines']
        if path.split('\\')[-2] == "train":
            RD_sequences = loader.readSequences(path)
        else:
            RD_sequences = sorted(loader.readSequences(path))
        global_train_mean = 1.6545123 
        global_train_std = 0.032430325
        count = 0
        a_id = 0
        while count < len(RD_sequences):
            objects = []
            RD_filename = RD_sequences[count]

            # Load GT instances
            gt_filename = loader.gtfileFromRDfile(RD_filename, path)
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path" ,gt_filename) #dodala sam gt_filename
            # Get RD spectrum
            # img[49:49 + 535, 97:97 + 577]
            RD_data = np.sum(loader.readRD(RD_filename)[49:49+535, 97:97+577, :], axis=2)
            #RD_data = ndimage.zoom(RD_data, [1 / 5, 1 / 5]) # 107 x 115
            RD_data = (RD_data - global_train_mean) / global_train_std
            # Get RD bboxes
            bboxes, classes = helper.readAndEncodeGtRD(gt_instances, RD_data.shape)
            seq_id = RD_filename.split('\\')[-1]#.split('_')[-1]
            for (box, class_) in zip(bboxes, classes):
                bbox, area = helper.buildTfdsBoxes(box)
                objects.append({
                    'bbox': bbox,
                    'label': classes_list.index(class_),
                    'area': area,
                    'id': a_id
                })
                a_id += 1
            image_filename = RD_filename#loader.imgfileFromRADfile(RAD_filename, path)
            example = {
                'spectrum': RD_data.astype(np.float32),
                'spectrum/filename': RD_filename,
                'sequence/id': count,#int(seq_id),
                'image': image_filename,
                'spectrum/id': count,
                'objects': objects
            }
            count += 1

            yield count, example
