import glob
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


def readSequences(path):
    """ Read sequences from PROJECT_ROOT/sequences.txt """
    sequences = glob.glob(os.path.join(path, "RD\\*.png"))
    if len(sequences) == 0:
        raise ValueError("Cannot read sequences.txt. \
                    Please check if the file is organized properly.")
    return sequences


def readRD(filename):
    if os.path.exists(filename):
        return np.array(plt.imread(filename))#np.load(filename)
    else:
        return None


def gtfileFromRDfile(RD_file, prefix):
    """ Transfer RD filename to gt filename """
    RD_file_spec = RD_file.split("RD")[-1]
    gt_file = os.path.join(prefix, "gt") + RD_file_spec.replace("png", "pickle")
    return gt_file


def imgfileFromRADfile(RAD_file, prefix):
    """ Transfer RAD filename to gt filename """
    RAD_file_spec = RAD_file.split("RAD")[-1]
    gt_file = os.path.join(prefix, "stereo_image") + RAD_file_spec.replace("npy", "jpg")
    return gt_file


def readRadarInstances(pickle_file):
    """ read output radar instances. """
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            radar_instances = pickle.load(f)
        if len(radar_instances['classes']) == 0:
            radar_instances = None
    else:
        radar_instances = None
    return radar_instances
