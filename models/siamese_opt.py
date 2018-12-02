import sys
import os
import argparse
import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization


new_path = os.getcwd() + '/src'
if new_path not in sys.path:
    sys.path.append(new_path)

from data.data_utils import BatchSampler, OneShotGenerator


parser = argparse.ArgumentParser()
parser.add_argument(
            "--path",
            help="path to store models at.",
            default=os.getcwd()
    )
parser.add_argument(
            "--batch_size",
            help="batch size for training.",
            default=128
    )
parser.add_argument(
            "--half_expand_factor",
            help="Half of the factor by which to grow the dataset in pairing.",
            default=5
    )
parser.add_argument(
            "--max_iter",
            help="Max number of iterations to run the bayesian optimization.",
            default=30
    )
args = parser.parse_args()
MODELS_DIR = args.path


################################################################################
####   Load in training and split to train, val sets.  Set samplers         ####
################################################################################

train_npzfile = np.load(os.getcwd() + '/data/processed/train.npz')
X_train, y_train = train_npzfile['arr_0'], train_npzfile['arr_1']
X_train = X_train.reshape(X_train.shape[0], 105, 105, 1)

# define batch size and the expansion factor for the dataset
batch_size = args.batch_size
half_expand_factor = args.half_expand_factor

y_train_pd = pd.DataFrame(data=y_train, columns=['Alphabet', 'Character', 'Drawer'])
drawers = y_train_pd.Drawer.unique()
trn_drawers = np.random.choice(drawers, 16, replace=False)
trn_inds = y_train_pd.Drawer.isin(trn_drawers)
X_trn, y_trn = X_train[trn_inds], y_train[trn_inds]
X_val, y_val = X_train[~trn_inds], y_train[~trn_inds]
X_trn.shape, y_trn.shape, X_val.shape, y_val.shape

TrainSampler = BatchSampler(X_trn, y_trn,
                            batch_size=batch_size,
                            half_expand_factor=half_expand_factor)
ValSampler = OneShotGenerator(X_val, y_val)

################################################################################
####   Define some utility functions for the the optimization process       ####
################################################################################

def build_siamese_net(dropout_conv,
                      dropout_fc,
                      learning_rate):

    input_shape = (105, 105, 1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64,(10,10), activation='relu',
                   input_shape=input_shape,
                   kernel_initializer=keras.initializers.VarianceScaling()))
    convnet.add(MaxPooling2D())
    convnet.add(Dropout(dropout_conv))
    convnet.add(Conv2D(128,(7,7), activation='relu',
                   kernel_initializer=keras.initializers.VarianceScaling()))
    convnet.add(MaxPooling2D())
    convnet.add(Dropout(dropout_conv))
    convnet.add(Conv2D(128,(4,4), activation='relu',
                   kernel_initializer=keras.initializers.VarianceScaling()))
    convnet.add(MaxPooling2D())
    convnet.add(Dropout(dropout_conv))
    convnet.add(Conv2D(256,(4,4), activation='relu',
                   kernel_initializer=keras.initializers.VarianceScaling()))
    convnet.add(Flatten())
    convnet.add(Dropout(dropout_fc))
    convnet.add(Dense(4096, activation="sigmoid"))

    # Encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # Merge the two outputs
    L1_distance = lambda x: K.abs(x[0]-x[1])
    both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
    prediction = Dense(1, activation='sigmoid')(both)
    siamese_net = Model(input=[left_input, right_input], output=prediction)

    optimizer = Adam(learning_rate)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)
    return siamese_net

def evaluate_model(model, sampler, X, m=2):
    accuracies = []
    for _ in range(m):
        preds = []
        (trn_idx, tst_idx), _ = sampler.generate_one_shot()
        n = trn_idx.shape[0]
        Trn_chars = X[trn_idx]
        for i in tst_idx:
            test_image = np.tile(X[i], (n, 1, 1, 1))
            preds.append(
                    np.argmax(model.predict([test_image, Trn_chars]))
                )
        accuracies.append(np.mean(preds == np.arange(n)))
    return np.mean(accuracies)

def train_siamese_net(siamese_net, epochs):
    epochs, count = epochs, 0

    for epoch in range(epochs):
        print("The current epoch is: {}".format(epoch))
        print("============================================================")
        accuracies = []
        for _ in range(TrainSampler.max_batches):
            pairs_b, y_b = TrainSampler.generate_batch()
            siamese_net.train_on_batch(pairs_b, y_b)
            if count % 50 == 0:
                accuracy = evaluate_model(siamese_net, ValSampler, X_val)
                print("Current estimate of 20-way accuracy is: {}".format(
                                    accuracy
                        ))
                accuracies.append(accuracy)
            count = (1 + count) % TrainSampler.max_batches
        print("============================================================")
        print("The accuracies for the last epoch were: ", accuracies)
        print("The average accuracy for the last epoch was: {}".format(
                    np.mean(accuracies)
            ))
        print("                                                            ")
        print("                                                            ")
    return np.mean(accuracies)

################################################################################
####   Set up the optimizer and run the optimization process.               ####
################################################################################

# Define parameter search space
bds = [
    {'name': 'dropout_conv', 'type': 'continuous', 'domain': (0, 0.5)},
    {'name': 'dropout_fc', 'type': 'continuous', 'domain': (0, 0.5)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-6, 1e-2)},
    {'name': 'epochs', 'type': 'continuous', 'domain': (50, 201)}
]

# Define optimization objective (make sure to save the model)
def trained_val_score(parameters):
    parameters = parameters[0]
    print("The current parameters are: ", 
                    str(np.around(parameters[0], decimals=6)) + ' ',
                    str(np.around(parameters[1], decimals=6)) + ' ',
                    str(np.around(parameters[2], decimals=6)) + ' ',
                    str(int(parameters[3]))
            )
    siamese_net = build_siamese_net(
                            parameters[0],
                            parameters[1],
                            parameters[2]
                    )
    score = train_siamese_net(siamese_net, int(parameters[3]))
    SAVE_PATH = MODELS_DIR + '/model' + '_' + \
                str(np.around(parameters[0], decimals=6)) + '_' + \
                str(np.around(parameters[1], decimals=6)) + '_' + \
                str(np.around(parameters[2], decimals=6)) + '_' + \
                str(int(parameters[3])) + '_' + \
                'score_' + str(score) + '.h5'
    siamese_net.save(SAVE_PATH)
    K.clear_session()
    return score

optimizer = BayesianOptimization(
                f=trained_val_score,
                domain=bds,
                model_type='GP',
                initial_design_numdata=10,
                acquisition_type='EI',
                maximize=True
    )

optimizer.run_optimization(max_iter=args.max_iter)
optimizer.save_report(report_file=MODELS_DIR+'/optimization_report')
