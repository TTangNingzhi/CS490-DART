import numpy as np
import pandas as pd
import torch.utils.data


class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

        self.num_users = np.max(self.items[:, 0]) + 1
        self.num_items = np.max(self.items[:, 1]) + 1
        self.__preprocess()
        self.noise_mask = np.zeros_like(self.targets, dtype=np.bool)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index], self.noise_mask[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

    def __preprocess(self):
        neg_items, neg_targets = self.__negative_sampling()
        self.items = np.concatenate((self.items, neg_items), axis=0)
        self.targets = np.ones_like(self.targets, dtype=np.float32)
        self.targets = np.concatenate((self.targets, neg_targets), axis=0)

    def __negative_sampling(self):
        num = 5 * len(self.targets)
        neg_users = np.random.randint(0, self.num_users, size=2 * num)
        neg_items = np.random.randint(0, self.num_items, size=2 * num)
        neg_features = np.concatenate((neg_users.reshape(-1, 1), neg_items.reshape(-1, 1)), axis=1)
        neg_features = np.array(list(set(map(tuple, neg_features)) - set(map(tuple, self.items))))
        neg_features = neg_features[:min(num, len(neg_features))]
        neg_targets = np.zeros(len(neg_features), dtype=np.float32)
        return neg_features, neg_targets

    def noisify(self, train_indices, noise_rate):
        neg_indices = np.where(self.targets == 0)[0]
        neg_train_indices = np.intersect1d(neg_indices, train_indices)
        noise_indices = np.random.choice(neg_train_indices, size=int(noise_rate * len(neg_train_indices)),
                                         replace=False)
        self.noise_mask[noise_indices] = True
        self.targets[noise_indices] = 1
        print("# of noise samples: {}".format(np.sum(self.noise_mask)))


class MovieLens1MDataset(MovieLens20MDataset):
    """
    MovieLens 1M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path, sep='::', engine='python', header=None)


class MovieLens100KDataset(MovieLens20MDataset):
    """
    MovieLens 100K Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path, sep='\t', engine='python', header=None)
