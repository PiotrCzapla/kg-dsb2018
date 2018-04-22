from pathlib import Path
import bcolz
import cv2
import numpy as np
import pandas as pd
import dlutils
from fastai.dataset import open_image

kg = dlutils.env.get_or_setup_competition("data-science-bowl-2018")

def setup_experiment(model_name):
    return dlutils.env.setup_kaggle(kg, model_name)

TRAIN_PATH = kg.input / 'stage1_train'
TEST_PATH = kg.input / 'stage1_test'