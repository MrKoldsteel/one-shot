import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Sampler
from PIL import Image
from torchvision import transforms


def find_omniglot_info(root_dir):
    """
    Walks the folders in root_dir and extracts relevant information
    about the omniglot dataset.  Specifically assumes that it is of
    the structure at https://github.com/brendenlake/omniglot, speci
    -fically that there is are images_background and images_evaluat
    -ion folders with sub-directories storing characters by alphabet.
    """
    Ext, Char, Dir, File, DSet, Cls, nc = [], [], [], [], [], {}, 0

    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if f.endswith("png"):
                Ext.append(f)
                r = root.split('/')
                lr = len(r)
                curr_char = r[lr - 2] + '/' + r[lr - 1]
                Char.append(curr_char)
                if curr_char not in Cls:
                    Cls[curr_char] = nc
                    nc += 1
                Dir.append(root)
                File.append(root + '/' + f)
                DSet.append(r[lr - 3])

    info = pd.DataFrame({
                'Ext': Ext,
                'Char': Char,
                'Dir': Dir,
                'File': File,
                'DSet': DSet
        })
    info['Cls'] = [Cls[x] for x in info.Char]
    info['Drawer'] = [x.split('.')[0].split('_')[1] for x in info.Ext]
    drawers = info.Drawer.unique()
    val_drawers = np.random.choice(drawers, 4, replace=False)
    trn_drawers = np.setdiff1d(drawers, val_drawers)
    info['Mode'] = 'tst'
    info.loc[(info.DSet == 'images_background') &
             info.Drawer.isin(val_drawers), 'Mode'] = 'val'
    info.loc[(info.DSet == 'images_background') &
             info.Drawer.isin(trn_drawers), 'Mode'] = 'trn'
    return info


# Define a vanilla transform to pass to the omniglot dataset
# Class for image augmentation on loading. Can be changed on
# Initialization.
ra = transforms.RandomAffine(15, translate=[0, 0.05],
                             scale=[0.8, 1.2], fillcolor=1)
rs = transforms.Resize((28, 28))
transform = transforms.Compose([ra, rs])


# Something to consider here is that we do the caching of pos and
# negative pairs beforehand, from info.  Then the dataset caller here
# can return a pair of images and a label (cls == cls).  We can easily
# control the fraction of positive to negative example pairs here.
# Not really sure here how torch would deal with the pairs in batch
# Sampling of indices. Seems that this could be a cleaner approach to
# what we're looking to do.

def form_neg_pair(info, mode='trn'):
    """
    Returns a list of 2 indices correspondign to index values
    of info which store different characters.
    """
    y = info[info.Mode == mode]
    curr_ind = y.Char.sample(1).index.values[0]
    curr_char = y.Char.loc[curr_ind]
    new_ind = y[~y.Char.isin([curr_char])].sample(1).index.values[0]
    return [curr_ind, new_ind]


def form_pos_pair(info, i, mode='trn'):
    """
    Returns a list of 2 indices, i and another index j corresponding
    to a location in info which is of the same character as i.
    """
    y = info[info.Mode == mode]
    char = info.Char.loc[i]
    drawer = info.Drawer.loc[i]
    new_ind = y[(~y.Drawer.isin([drawer])) &
                y.Char.isin([char])].sample(1).index.values[0]
    return [i, new_ind]


def form_siamese_cache(info, cache_size=120000, neg_per_pos=5, mode='trn'):
    """
    Forms a cache of indices for the dataset object to call so we can
    load pairs.  neg_per_pos indicates, roughly, the number of negative
    pairs we would like per positive pair.  Seems this should be a tuning
    parameter.
    """
    cache = []
    effective_cache_size = int(np.ceil(
                         neg_per_pos * cache_size / (neg_per_pos + 1)))
    for i in range(effective_cache_size):
        neg1, neg2 = form_neg_pair(info, mode=mode)
        cache.append([neg1, neg2])
        if i % neg_per_pos == 0: # and i >= neg_per_pos:
            pos1, pos2 = form_pos_pair(info, neg1, mode=mode)
            cache.append([pos1, pos2])
    return cache


class OmniglotDataset(Dataset):
    """
    Create a class to take advantage of the pytorch dataloader.
    """

    def __init__(self, root_dir, cache_size, neg_per_pos,
                 mode='trn', augment=True, transform=transform):
        self.info = find_omniglot_info(root_dir)
        self.cache_size = cache_size
        self.cache = {}
        self.npp = neg_per_pos
        self.mode = mode
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return self.cache_size

    def cache_out(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        pos = np.random.binomial(1, 1 / self.npp)
        if pos and self.cache:
            idy = np.random.choice(list(self.cache.keys()))
            pos1, pos2 = form_pos_pair(self.info,
                                       self.cache[idy][0],
                                       mode=self.mode)
            self.cache[idx] = [pos1, pos2]
            return [pos1, pos2]
        neg1, neg2 = form_neg_pair(self.info, mode=self.mode)
        self.cache[idx] = [neg1, neg2]
        return [neg1, neg2]

    def __getitem__(self, idx):
        id1, id2 = self.cache_out(idx)
        img1_name = self.info.File.iloc[id1]
        img2_name = self.info.File.iloc[id2]
        image1 = Image.open(img1_name)
        image2 = Image.open(img2_name)
        # image = Image.open(img_name, mode='r').convert('L')
        if self.augment:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        # return np.array(image, dtype=int), self.info.Cls[idx]
        X1 = 1 - np.array(image1, dtype=int)
        X2 = 1 - np.array(image2, dtype=int)
        y = int(self.info.Cls[id1] == self.info.Cls[id2])
        return X1, X2, y


# Maybe the best place to form cache is outside of the Siamese
# Network.  Then the logic can be kept simpler inside. And it
# May be easier to test that things are working properly as this
# Class is then just referencing something that is easier. Can't
# Forget that we want three modes for the sampler:
# Training, Validation and Testing and these draw from different
# Portions of the dataset.

# Something to keep in mind here is that the fraction of positive
# to negative examples per character could be considered a tuning
# parameter.  For any support set in the 20-way one-shot task,
# there is only going to be one positive match, so there is nothing
# to say that one positive and one negative example of a character
# for every negative one is the way to get optimal training
# results for the 20-way task (probably better as we get closer
# to 1-way one-shot as now the training structure would more closely
# model the test time data.

# So might want to consider this as a parameter in get_siamese_batch.

# def get_same_char(info, mode='trn'):
#     """
#     Picks randomly selects a character and three different
#     versions of that character.  Two to be used for a matching
#     pair and the third to be used for a non-matching pair.
#     Returns the info indices of these.
#     """
#     y = info[info.Mode == mode]
#     char = np.random.choice(y.Char.unique())
#     return y[y.Char.isin([char])].sample(3).index.values


# def get_different_char(info, i, mode='trn'):
#     """
#     Picks a character from mode of different type than that in
#     info.Char.iloc[i].
#     """
#     y = info[info.Mode == mode]
#     curr_char = info.Char.iloc[i]
#     return y[~y.Char.isin([curr_char])].sample(1).index.values[0]


# def get_siamese_batch(info, batch_size=64, mode='trn'):
#     pos_inds, neg_inds = [], []
#     for _ in range(batch_size):
#         pos1, pos2, neg1 = get_same_char(info, mode=mode)
#         neg2 = get_different_char(info, neg1, mode=mode)
#         pos_inds.append([pos1, pos2])
#         neg_inds.append([neg1, neg2])
#     return np.array(pos_inds), np.array(neg_inds)

# Maybe the easiest thing to do here is use get siamese_batch
# to form a siamese cache, which is composed of these siamese
# batches of indices.  We can then have a positive and negative
# sampler that return iters over these.

# One thing that this would mean is that mode='trn','val' or 'tst'
# Would have to be specified on formation of the cache, and then
# and then a different sampler object created for each cache.


# def form_siamese_cache(info, batch_size=64, mode='trn', cache_size=1000):
#     cache = {}
#     for i in range(cache_size):
#         cache[i] = get_siamese_batch(info, batch_size=batch_size, mode=mode)
#     return cache
