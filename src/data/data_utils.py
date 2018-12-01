import numpy as np
import pandas as pd


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
                (self.X[X_b_inds[:, 0], :],
                 self.X[X_b_inds[:, 1], :]),
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
        self.form_cache()

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
                    (test_inds, train_inds),
                    (self.y.iloc[test_inds],
                     self.y.iloc[train_inds])
        )

    def form_cache(self):
        for i in range(self.cache_size):
            self.cache[i] = self.form_one_shot()

    def generate_one_shot(self):
        if self.mode == 'cached':
            current_one_shot = self.cache[self.current_task_number]
            self.current_task_number = (
                    (1 + self.current_task_number) % self.cache_size
                )
            return current_one_shot
        return self.form_one_shot()
