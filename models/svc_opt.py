import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.decomposition import TruncatedSVD, MiniBatchDictionaryLearning
from imgaug import augmenters as iaa
from sklearn.svm import SVC
from GPyOpt.methods import BayesianOptimization
import pickle

new_path = os.getcwd() + '/../src'
if new_path not in sys.path:
    sys.path.append(new_path)

from data.data_utils import OneShotGenerator


################################################################################
######    Load in the data, make preliminary transformations      ##############
################################################################################


# Load the background dataset. X_background contains the background
# images and y_background class information.
print("                                                                ")
print("Loading background images and labels.")
print("-------------------------------------")
train_npz_file = np.load('../data/processed/train.npz')
X_background, y_background = train_npz_file['arr_0'], train_npz_file['arr_1']
print("X_background shape: ", X_background.shape)
print("y_background shape: ", y_background.shape)
print("                                                                ")
print("================================================================")
print("                                                                ")

# Load the evaluation dataset. X_evaluation contains the background
# images and y_evaluation class information.
print("Loading evaluation images and labels.")
print("-------------------------------------")
test_npz_file = np.load('../data/processed/test.npz')
X_evaluation, y_evaluation = test_npz_file['arr_0'], test_npz_file['arr_1']
print("X_evaluation shape: ", X_evaluation.shape)
print("y_evaluation shape: ", y_evaluation.shape)
print("                                                                ")
print("================================================================")
print("                                                                ")

# Put the class information in pandas arrays for help separating training
# and validation data below
y_background_pd = pd.DataFrame(
                    data=y_background,
                    columns=['Alphabet', 'Character', 'Drawer']
    )
y_evaluation_pd = pd.DataFrame(
                    data=y_evaluation,
                    columns=['Alphabet', 'Character', 'Drawer']
    )

print("Splitting background set into training and validation sets.")
print("-----------------------------------------------------------")
drawers = y_background_pd.Drawer.unique()
trn_drawers = np.random.choice(drawers, 16, replace=False)
trn_inds = y_background_pd.Drawer.isin(trn_drawers)
X_trn, y_trn = X_background[trn_inds], y_background[trn_inds]
X_val, y_val = X_background[~trn_inds], y_background[~trn_inds]
print("X_trn shape: ", X_trn.shape)
print("y_trn shape: ", y_trn.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)
print("                                                                ")
print("================================================================")
print("                                                                ")


# Helper function for image augmentation
# Expands a batch of images by including random rotations, translations,
# etc.  Augmenting things to mimic natural variations in handwriting.
aug = iaa.Affine(scale=(0.8, 1.2), translate_px=(-5, 5),
                 rotate=(-18, 18), cval=1)

def augment_it_on_batch(x, y, n=5):
    n_way = x.shape[0]
    new_x = np.copy(x)
    for i in range(n):
        new_x = np.concatenate(
                (new_x, aug.augment_images(x).reshape(n_way, 105, 105))
                )
    return new_x, np.tile(y, n + 1)


# Form one-shot task generators from training, validation and evaluation
# data sets.
TrnData = OneShotGenerator(X_trn, y_trn)
ValData = OneShotGenerator(X_val, y_val)
TstData = OneShotGenerator(X_evaluation, y_evaluation)

################################################################################
########         Setup models and evaluation functions       ###################
################################################################################

# PCA and TruncatedPCA are linear methods, so we should be able to fit to the
# Highest dimension and then retain what we want. Not obvious whether or not
# This is the case with the others.
trans = {'mbdl': MiniBatchDictionaryLearning,
         'tsvd': TruncatedSVD,
         'pca': PCA,
         'kpca': KernelPCA}
dims = [10, 15, 20, 25, 30]

print("Fitting the dimention reduction transforms to training data.")
print("------------------------------------------------------------")
transforms = {}
for key, val in trans.items():
    for dim in dims:
        print("Initializing transform {}".format(key + '_' + str(dim)))
        transforms[key + '_' + str(dim)] = val(n_components=dim)
        transforms[key + '_' + str(dim)].fit(
                        X_trn.reshape(X_trn.shape[0], -1)
                    )
print("                                                                ")
print("================================================================")
print("                                                                ")

n_models = len(transforms.keys())
# Setup the bayesian optimization for svc and then loop through transforms
# and bayesian optimize for each transform.  Save the optimization outputs
# in a dictionary and pickle it.

# A function to evaluate a classification model of the scikit api (fit,
# score).
def eval(clf, trans, data=ValData, n_evals=20, n_reps=20):
    accuracies = []
    for _ in range(n_evals):
        # Generate tasks from validation data
        x_pairs, y_pairs = data.generate_one_shot()
        n_way = x_pairs[0].shape[0]
        x, y = augment_it_on_batch(x_pairs[1], np.arange(n_way), n=n_reps)

        # Transform the data
        trn_features = trans.transform(x.reshape(n_way * (n_reps + 1), -1))
        tst_features = trans.transform(x_pairs[0].reshape(n_way, -1))

        # Fit classifier to the training data, predict and
        # evaluate accuracy.
        clf.fit(trn_features, y)
        accuracies.append(clf.score(tst_features, np.arange(n_way)))
    return np.mean(accuracies)

# Parameters to perform bayesian optimization over in search for optimal SVC
# hyperparameters.
parameters = [{'name': 'C',      'type': 'continuous', 'domain': (0.0001, 100)},
              {'name': 'gamma',  'type': 'continuous', 'domain': (0.0001, 100)}]

def svr_val(trans):
    def f(params):
        params = params[0]
        clf = SVC(C=params[0], kernel='poly', gamma=params[1])
        return eval(clf, trans)
    return f

print("Running optimization on the models.")
print("================================================================")
# Define a dictionary to store the results of the Bayesian optimization
results, count = {}, 0
for key, val in transforms.items():
    print("Running optimization for model: {}".format(key))
    print("This is {} of {} optimizations.".format(count + 1, n_models))
    print("---------------------------------")
    opt = BayesianOptimization(f=svr_val(val),
                               domain=parameters,
                               initial_design_numdata=30,
                               num_cores=10,
                               maximize=True)
    opt.run_optimization(max_iter=30)
    results[key] = opt.get_evaluations()
    # Pickle the results so we can go through and find the best model. Do This
    # in loop so that we can walk away with results pre completion if needed.
    with open('svc_opt_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Update count
    count += 1
