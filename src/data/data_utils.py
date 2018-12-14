import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


class BatchSampler:
    """
    Creates a batch sampler from raw-ish data pairs X, y.  The data needed for
    training a siamese networks are image pairs from X and binary indicators
    indicating whether they are equal.  Instead of forming these pairs
    explicitly, this sampler creates a dictionary with keys corresponding to
    the rows of X. The item stored at each key consists of a tuple with:
        - a list of indices of rows of X to pair with the current key
        - item a list of binary indicators inidcating whether they are of the
          same type.
    Although this takes a while to initialize (as initialization calls this
    pair forming operation), this results in fairly efficient data storage,
    access and batch construction.

    Attributes (this should probably be the actual attributes, not the entries
    to __init__, which should be in the document there)
    ----------
    X : array
        an n * h * w * n_channels array containing images, presumably from the
        dataset we want to generate batches for (presumably for a siamese net
        training task).
    y : array
        an n * 3 array containing the specifics of the characters in X.  The
        columns are 'Alphabet', 'Character', 'Drawer'.  This makes this class
        quite specific to omniglot.
    batch_size : integer
        this represents the size of the batches we ultimately want to generate.
    half_expand_factor : integer
        this represents half of the size by which we would like to expand the
        dataset, or half of the number of pairs we would like to construct for
        each row of X.

    Methods
    -------
    form_pairs()
        forms the dictionary (this should probably be a private method as it is
        called on initialization and not meant to be used.  Then it doesn't
        enter into the docstring here or possibly at all).
        into the docstring here or possibly at all).

    This is the general idea for forming these docstrings.  Description then
    outline Attributes and public Methods.  Functions inside then have regular
    style docstrings.

    The way sampling is done here means we probably end up passing the same
    pairs in here and there, but hopefully this won't hold things up too much.
    """

    def __init__(self, X, y, batch_size=64, half_expand_factor=4):
        self.X = X
        self.y_n_pd = y
        self.y = pd.DataFrame(
                        data=y,
                        columns=['Alphabet', 'Character', 'Drawer']
                    )
        self.B = batch_size
        self.E = half_expand_factor
        self.n = y.shape[0]
        self.pairs = self.form_pairs()
        self.inds = np.random.permutation(y.shape[0])
        self.current_batch = 0
        self.current_efactor = 0
        self.max_batches = 1 + y.shape[0] // batch_size

    def form_pairs(self):
        pairs = {}
        for i in range(self.n):
            al, ch, dr = self.y.iloc[i]
            same_inds = self.y[
                    (self.y.Alphabet == al) &
                    (self.y.Character == ch) &
                    (self.y.Drawer != dr)
                ].sample(self.E, replace=False).index.values
            diff_inds = self.y[
                   ~((self.y.Alphabet == al) &
                     (self.y.Character == ch))
                ].sample(self.E, replace=False).index.values
            curr_y = np.zeros(2 * self.E)
            curr_y[:self.E] = 1
            p = np.random.permutation(2 * self.E)
            pairs[i] = (
                    np.concatenate((same_inds, diff_inds))[p],
                    curr_y[p]
                )
        return pairs

    def generate_batch(self):
        """
        Needs to be written.
        """
        current_inds = self.inds[
                    range(
                        self.current_batch * self.B,
                        min((1 + self.current_batch) * self.B, self.n)
                    )
        ]
        X_b_inds, y_b = [], []
        for ind in current_inds:
            X_b_inds.append((ind, self.pairs[ind][0][self.current_efactor]))
            y_b.append(self.pairs[ind][1][self.current_efactor])
        X_b_inds = np.array(X_b_inds)
        self.current_batch = (1 + self.current_batch) % self.max_batches
        if self.current_batch == 0:
            self.current_efactor = (1 + self.current_efactor) % (2 * self.E)
        return (
                [self.X[X_b_inds[:, 0], :],
                 self.X[X_b_inds[:, 1], :]],
                np.array(y_b)
            )

    def generate_one_shot(self, n=20):
        """
        Although this could also be a class that subclasses this.
        """
        alphabet = np.random.choice(self.y.Alphabet.unique())
        drawers = np.random.choice(
                        self.y.Drawer.unique(),
                        2, replace=False
                    )
        characters = np.random.choice(
                        self.y[
                            self.y.Alphabet == alphabet
                        ].Character.unique(),
                        n, replace=False
                    )

        test_inds = self.y[
                        (self.y.Alphabet == alphabet) &
                        (self.y.Drawer == drawers[0]) &
                        np.isin(self.y.Character, characters)
                    ].index.values

        train_inds = self.y[
                        (self.y.Alphabet == alphabet) &
                        (self.y.Drawer == drawers[1]) &
                        np.isin(self.y.Character, characters)
                    ].index.values
        return (
                    (test_inds, train_inds),
                    (self.y.iloc[test_inds],
                     self.y.iloc[train_inds])
        )


class OneShotGenerator:
    """
    A class that generates one-shot tasks.  It does this in two
    ways.
    - Caches a number of one-shot tasks and cycles through these.
      this is the default mode.
    - Randomly generates a one-shot task.  This is the mode that
      should be used on test data, but is a little slow for the
      purposes of model evaluation during training.
    Also, the default is an 10-way one-shot task, because this
    will be used on the training data, which has some alphabets
    which have a small number of characters (I think 11 or 12 is
    the smallest).  All alphabets in the test set have at least
    20 characters, so 20-way evaluation here is fine.
        We could also default to 20-way and if an alphabet only
    has m < 20 characters, then we fall back on an m-way challenge.
    This might actually be better.
    """
    def __init__(self, X, y, mode='cached', n_way=20, cache_size=320):
        self.X = X
        self.y_n_pd = y
        self.y = pd.DataFrame(
                        data=y,
                        columns=['Alphabet', 'Character', 'Drawer']
                    )
        self.mode = mode
        self.n_way = n_way
        self.cache_size = cache_size
        self.cache = {}
        # self.form_cache()

        self.n = y.shape[0]
        self.current_task_number = 0

    def form_one_shot(self):
        """
        Although this could also be a class that subclasses this.
        """
        alphabet = np.random.choice(self.y.Alphabet.unique())
        # make sure that the number of tasks that we don't attempt
        # to generate an n-way task when the alphabet is too small.
        n = min(
                self.n_way,
                self.y.Character[
                    self.y.Alphabet == alphabet
                ].unique().shape[0]
            )

        drawers = np.random.choice(
                        self.y.Drawer.unique(),
                        2, replace=False
                    )
        characters = np.random.choice(
                        self.y[
                            self.y.Alphabet == alphabet
                        ].Character.unique(),
                        n, replace=False
                    )

        test_inds = self.y[
                        (self.y.Alphabet == alphabet) &
                        (self.y.Drawer == drawers[0]) &
                        np.isin(self.y.Character, characters)
                    ].index.values

        train_inds = self.y[
                        (self.y.Alphabet == alphabet) &
                        (self.y.Drawer == drawers[1]) &
                        np.isin(self.y.Character, characters)
                    ].index.values
        return (
                    (self.X[test_inds], self.X[train_inds]),
                    (self.y.iloc[test_inds],
                     self.y.iloc[train_inds])
        )

    def form_cache(self):
        for i in range(self.cache_size):
            self.cache[i] = self.form_one_shot()

    def generate_one_shot(self):
        if self.mode == 'cached':
            if self.current_task_number not in self.cache:
                current_one_shot = self.form_one_shot()
                self.cache[self.current_task_number] = current_one_shot
            else:
                current_one_shot = self.cache[self.current_task_number]
            self.current_task_number = (
                    (1 + self.current_task_number) % self.cache_size
                    )
            return current_one_shot
        return self.form_one_shot()


class BatchingForMatching:

    def __init__(self, X, y, mode='cache', cache_size=20,
                       batch_size=32, n_way=20):
        self.X = X
        self.c = pd.DataFrame(
                        data=[ch[0] + '_' + ch[1] for ch in y],
                        columns=['Character']
                    )
        self.unique_characters = np.random.permutation(
                                    self.c.Character.unique())
        self.num_uc = self.unique_characters.shape[0]
        self.batch_num = 0 # keep track of current batch number
        self.curr_cache = 0 # and current position in the cache dict
        self.y = pd.DataFrame(
                        data=y,
                        columns=['Alphabet', 'Character', 'Drawer']
                    )
        self.mode = mode
        self.n_way = n_way
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.cache = {}
        # self.form_cache()

        self.n = X.shape[0]
        self.current_task_number = 0

    def get_one_shot_indices(self, ch): #,  y=y_train_pd, c=characters, bs=32):
        """
        inputs a character and generates a one-shot task for it
        by selecting two random drawers and 19 other characters
        to form a one-shot task around.  This is all done in terms
        of indices.  We can then use this to store a large number
        of one-shot tasks for each character.
        """
        tst_inds, trn_inds = [], []
        for _ in range(self.batch_size):
            rnd_dr = np.random.choice(self.y.Drawer.unique(),
                                  2, replace=False)
            rnd_ch = np.random.choice(
                    self.c.Character[self.c.Character != ch].unique(),
                    19, replace=False)
            rnd_ch = np.append(ch, rnd_ch)
            c_trn_inds = self.y[(self.c.Character.isin(rnd_ch)) &
                       (self.y.Drawer == rnd_dr[0])].index.values
            c_tst_inds = self.y[(self.c.Character == ch) &
                       (self.y.Drawer == rnd_dr[1])].index.values
            cls = np.argmax(
                self.c.Character.iloc[c_trn_inds].isin([ch]).values
                )
            tst_inds.append((c_tst_inds[0], cls))
            trn_inds.append(c_trn_inds)
            # print(y.iloc[c_tst_inds])
            # print(y.iloc[c_trn_inds[0]])
        return np.array(tst_inds), np.array(trn_inds)

    def form_cache(self):
        for ch in self.unique_characters:
            self.cache[ch] = []
            for _ in range(self.cache_size):
                self.cache[ch].append(self.get_one_shot_indices(ch))

    def generate_batch(self):
        """
        This is more or less giving what we wanted.  Now, the thing
        to do is to adjust the output so that it is putting out the
        torch tensors that we want for training...  So we need to
        form these from what is currently out, and then output them
        instead.  Cool beans.
        """
        if self.unique_characters[self.batch_num] not in self.cache:
            self.cache[
                    self.unique_characters[self.batch_num]
                ] = []
            self.cache[
                    self.unique_characters[self.batch_num]
                ].append(self.get_one_shot_indices(
                             self.unique_characters[self.batch_num]
                                       ))
        elif len(self.cache[self.unique_characters[self.batch_num]]) < self.cache_size:
            self.cache[
                    self.unique_characters[self.batch_num]
                ].append(self.get_one_shot_indices(
                             self.unique_characters[self.batch_num]
                                       ))
        out = self.cache[
                    self.unique_characters[self.batch_num]
                ][self.curr_cache]
        self.batch_num = (1 + self.batch_num) % self.num_uc
        if self.batch_num == 0:
            self.curr_cache = (1 + self.curr_cache) % self.cache_size

        tst_inds, trn_inds = out

        # Construct torch tensors for target images and classes
        target_ims = torch.tensor(self.X[tst_inds[:, 0]])
        target_classes = torch.tensor(tst_inds[:, 1])
        # target_classes = np.zeros((self.batch_size, self.n_way))
        # target_classes[np.arange(self.batch_size), tst_inds[:, 1]] = 1
        # target_classes = torch.tensor(target_classes, dtype=torch.int32)

        # Construct torch tensors for support set images and classes
        support_set_ims = torch.stack(
                [torch.tensor(self.X[trn_inds[i, :]]) for i in range(self.batch_size)]
            )
        support_set_classes = torch.stack(
                [torch.eye(self.n_way) for i in range(self.batch_size)]
            )
        return target_ims, target_classes, support_set_ims, support_set_classes


def concat_images(X):
    """Concatenates a bnch of images into a big matrix for plotting purposes."""
    nc,h,w,_ = X.shape
    X = X.reshape(nc,h,w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img, n

def plot_oneshot_task(pairs):
    """Takes a one-shot task given to a siamese net and  """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 12))
    fig.set_facecolor('white')
    ax1.set_title("Test Images")
    ax2.set_title("Support Set")
    #ax1.matshow(pairs[0][0].reshape(105,105),cmap='gray')
    img1, n = concat_images(pairs[0])
    ax1.matshow(img1,cmap='gray')
    img2, n = concat_images(pairs[1])
    ax2.matshow(img2,cmap='gray')
    plt.xticks(np.arange(0,105*n,105))
    plt.yticks(np.arange(0,105*n,105))
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax1.set_ylim(bottom=420, top=0)
    ax2.set_ylim(bottom=420, top=0)
    # ax2.grid(linewidth=1,linestyle='-',color='black')
    plt.show()
