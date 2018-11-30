# -*- coding: utf-8 -*-
import sys
import numpy as np
from matplotlib.image import imread
import pickle
import os
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument(
            "--path",
            help="Path to omniglot data folder"
    )
parser.add_argument(
            "--save",
            help="Path to save processed data to",
            default=os.getcwd()
    )

args = parser.parse_args()

RAW_DATA_PATH = os.path.join(args.path)
PROCESSED_DATA_PATH = args.save
TRAIN_DATA_FOLDER = os.path.join(RAW_DATA_PATH, 'images_background')
TEST_DATA_FOLDER = os.path.join(RAW_DATA_PATH, 'images_evaluation')


def process_data(path_from, path_to):
    """
    Processes the data residing in path, where path has a tree structure. At the
    first level are languages, and at the second level, characters.  In the char
    -acter folders are multiple renderings of the given character by various art
    -ists stored as images in .png format.  This function loads these, in image
    format into a numpy array X and keeps track of the language and character
    that the rows of X belong to (stored in y).

    Parameters
    ----------
    path : string
        the path to the folder described above.

    Returns
    -------
    X : array
        n * 105 * 105 * 1 array, with row i containing the data for image i and
        n is the number of images in the folder at path.
    y : array
        n * 2 array, with row i containing the alphabet and letter (in strings)
        that the image in row i of X belongs to.
    """
    X = []
    y = []
    for alphabet in os.listdir(path_from):
        print("Loading alphabet: {}".format(alphabet))
        alphabet_path = os.path.join(path_from, alphabet)
        for character in os.listdir(alphabet_path):
            letter_path = os.path.join(alphabet_path, character)
            for rendering in os.listdir(letter_path):
                drawer = rendering.split('_')[1].split('.')[0]
                y.append((alphabet, character, drawer))
                image = imread(
                            os.path.join(letter_path, rendering)
                    )
                X.append(image)
    np.savez(path_to, np.array(X), np.array(y))


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    process_data(
            TRAIN_DATA_FOLDER,
            os.path.join(PROCESSED_DATA_PATH, 'train')
        )
    process_data(
            TEST_DATA_FOLDER,
            os.path.join(PROCESSED_DATA_PATH, 'test')
        )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
