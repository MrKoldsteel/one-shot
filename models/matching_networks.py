import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime as dt

# Add utilities folder to path and load in what we need
new_path = os.getcwd() + '/../src'
if new_path not in sys.path:
    sys.path.append(new_path)

from data.pytorch_utils import MatchingOmniglotDataset, find_omniglot_info

# Set data type and device to use.
USE_GPU, dtype = True, torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# Path to Omniglot
data_dir = '../data/raw'
writer = SummaryWriter(comment='matching_nets')
# can acces date with dt.now().isoformat() which is a
# string looking like '2018-12-16T17:00:10.684090' so
# if we do dt.now().isoformat().split(':')[0], we should
# get a good date identifier to start off a filename
# with.

# How to call the MatchingOmniglotDataset class.
# MatchingOmniglotDataset(data_dir, n_way=20, cache_size=10000,
#                         transform=transform, mode='trn')


# Setup the model from the matching networks paper
# Define a utility class to flatten a layer.
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


# The next two functions define piece together the encoder
# of the paper, used to create a distribution over nearest
# neighbors in the support set.
def convLayer(in_planes, out_planes, useDropout = False):
    "3x3 convolution with padding"
    seq = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    if useDropout: # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq

encoder = torch.nn.Sequential(
                convLayer(1, 64),
                convLayer(64, 64),
                convLayer(64, 64),
                convLayer(64, 64),
                Flatten()
    )


# The next two functions make predictions given an encoder, a target image
# Support set images and one-hot encodings of the support set images.  The
# Output is a distribution over the the classes.
def get_preds(target_ims, support_set_ims, support_set_classes, model):
    # Calculate target and support set embeddings by model
    target_embedding = model(target_ims)
    n_b = support_set_classes.shape[0]
    n_ss = support_set_classes.shape[-2]
    d_embedding = target_embedding.shape[-1]
    support_set_embeddings = torch.stack(
        [model(support_set_ims[:, i, :, :, :]) for i in range(n_ss)], dim=1
        )

    # calculate similarities, softmax_sims and output predictions
    similarities = cos(
            target_embedding.reshape(n_b, 1, d_embedding).repeat([1, n_ss, 1]),
            support_set_embeddings
        )  #.t()
    # This may have been why it wasn't training properly: CrossEntropyLoss
    # takes scores! As does BCE... This may have been why the problems on
    # colab as well.
    # softmax_sims = softmax(similarities).unsqueeze(1)  #.t()).unsqueeze(1)
    # return softmax_sims.bmm(support_set_classes).squeeze()
    return similarities.unsqueeze(1).bmm(support_set_classes).squeeze()


# A utility to check the accuracy of method during training.
#def check_accuracy(data, model):
#    # See if this works if we just assume loader needs to be unpacked,
#    # as the data in the loop below
#    model.eval()  # set model to evaluation mode
#    with torch.no_grad():
#        target_ims, target_classes, support_set_ims, ss_one_hot = data
#        # Push batch to device
#        target_ims = target_ims.to(device=device, dtype=dtype)
#        target_classes = target_classes.to(device=device, dtype=torch.long)
#        support_set_ims = support_set_ims.to(device=device, dtype=dtype)
#        ss_one_hot = ss_one_hot.to(device=device, dtype=dtype)
#
#        # Push batch through the model
#        preds = get_preds(target_ims, support_set_ims, ss_one_hot, model)
#        acc = (torch.argmax(preds, dim=1) == target_classes).mean()
#        # print('Got %d / %d correct (%.2f)' % (num_correct, tbs , 100 * acc))
#        return acc


# Now setup the dataset loaders and loop through printing out and saving info
# to the SummaryWriter.

# We'll define a scheduler that outputs nway size of the experiment given the
# current epoch (seems helpful when we start with smaller numbers and finish
# big).  We train the model to learn on progressively harder n-way tasks. we
# can also vary this up within epochs
class NwayScheduler:
    def __init__(self, epoch_ranges=np.array([[0, 10], [10, 20], [20, 30], [30, 40], [40, np.inf]]),
                       n_way_bounds=np.array([[5,  10], [8,   15], [12,  18], [15, 20],  [20, 21]])):
        self.ers = epoch_ranges
        self.nws = n_way_bounds
        self.n = len(self.nws)
    def get_n_way(self, epoch):
        row_num = min(np.sum(epoch > self.ers[:, 1]), self.n - 1)
        low, high = self.nws[row_num, :]
        return np.random.randint(low, high)

# Utility to unpack and push variables to device.
def unpack_data(data):
    target_ims, target_classes, ss_ims, ss_classes = data
    target_ims = target_ims.to(device=device, dtype=dtype)
    target_classes = target_classes.to(device=device, dtype=torch.long)
    ss_ims = ss_ims.to(device=device, dtype=dtype)
    ss_classes = ss_classes.to(device=device, dtype=dtype)
    return target_ims, target_classes, ss_ims, ss_classes


# A utility to check the accuracy of method during training.
def check_accuracy(data, model):
    # See if this works if we just assume loader needs to be unpacked,
    # as the data in the loop below
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        target_ims, target_classes, ss_ims, ss_one_hot = unpack_data(data)
        # Push batch through the model
        preds = get_preds(target_ims, ss_ims, ss_one_hot, model)
        model_acc = (torch.argmax(preds, dim=1) == target_classes).cpu().numpy().mean()
        model_loss = loss(preds, target_classes).item()
        # print('Got %d / %d correct (%.2f)' % (num_correct, tbs , 100 * acc))
    return model_acc, model_loss




# Better not to form info in the initialization of the class, if we'll be
# initializing it multiple times on the same set of info.
print("                                               ")
print("Loading folder info and setting up data objects")
print("                                               ")
info = find_omniglot_info(data_dir)
val = MatchingOmniglotDataset(info, n_way=20, mode='val')
tst = MatchingOmniglotDataset(info, n_way=20, mode='tst')
val_loader = DataLoader(val, batch_size=32, shuffle=True, num_workers=3)
tst_loader = DataLoader(tst, batch_size=32, shuffle=True, num_workers=3)


optimizer = optim.Adam(encoder.parameters(), lr=1e-4)
encoder = encoder.to(device=device)

epochs = 1000
running_loss = 0
running_accuracy = 0
effective_i = 0
losses = []
print_every = 50
best_val_acc = 0
best_tst_acc = 0

nWays = NwayScheduler()

loss = torch.nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=2)
softmax = torch.nn.Softmax(dim=1)

val_loss, val_acc, tst_loss, tst_acc = 0, 0, 0, 0

print("Beginning training \n")

for epoch in range(epochs):
    # Change the n_way of the training data every four epochs to make the
    # learning task gradually harder.
    if epoch % 4 == 0:
        n_way = nWays.get_n_way(epoch)
        trn = MatchingOmniglotDataset(info, n_way=n_way, mode='trn')
        trn_loader = DataLoader(trn, batch_size=32, shuffle=True, num_workers=3)
    print("In epoch {}. The current N-way task is: {} \n".format(epoch, n_way))
    print("=================================================================== \n")
    writer.add_scalar('current N-way', n_way, effective_i)

    loader = zip(trn_loader, val_loader, tst_loader)
    for i, (train, validation, test) in enumerate(loader):
        # Set the encoder to train
        encoder.train()

        # Unpack training data and push to device (unpack does this) make
        # predictions, calculate loss and push back through the graph
        target_ims, target_classes, ss_ims, ss_classes = unpack_data(train)
        preds = get_preds(target_ims, ss_ims, ss_classes, encoder)
        lp = loss(preds, target_classes)
        optimizer.zero_grad()
        lp.backward()
        optimizer.step()

        # Calculate loss and accuracy to pass to writer.
        current_loss = lp.item()
        current_accuracy = (torch.argmax(preds, dim=1) == target_classes).cpu().numpy().mean()
        # running_loss += current_loss
        # running_accuracy += current_accuracy

        writer.add_scalars('train_metrics', {'training loss': current_loss,
                                             'training accuracy': current_accuracy},
                                             effective_i)

        if (i + 1) % print_every == 0:
            # Evaluate losses and accuracies on train and test batches
            # Also, save best model.
            val_acc, val_loss = check_accuracy(validation, encoder)
            tst_acc, tst_loss = check_accuracy(test, encoder)
            if val_acc > best_val_acc:
                file_name = dt.now().isoformat().split(':')[0]
                file_name += '_match_net_val_' + str(val_acc) + '_' + str(tst_acc) + '.pt'
                torch.save(encoder, file_name)
                best_val_acc = val_acc
            if tst_acc > best_tst_acc:
                file_name = dt.now().isoformat().split(':')[0]
                file_name += '_match_net_tst_' + str(val_acc) + '_' + str(tst_acc) + '.pt'
                torch.save(encoder, file_name)
                best_tst_acc = tst_acc


        writer.add_scalars('test_metrics', {'validation accuracy': val_acc,
                                            'validation loss': val_loss,
                                            'testing accuracy': tst_acc,
                                            'testing loss': tst_loss},
                                                 effective_i)

        effective_i += 1
