import numpy as np
import pandas as pd
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, sep='\t', engine='c', header='infer', num_neg=1):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        print("********** stats of dataset **********")
        self.pos_items = data[:, :2]
        self.noise_masks = ~data[:, 2].astype(np.bool)
        print("num of tp:", np.sum(~self.noise_masks), "fp:", np.sum(self.noise_masks))
        self.pos_targets = np.ones_like(self.noise_masks, dtype=np.float32)

        self.field_dims = np.max(self.pos_items, axis=0) + 1
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)
        self.num_users = np.max(self.pos_items[:, 0]) + 1
        self.num_items = np.max(self.pos_items[:, 1]) + 1

        self.num_neg = num_neg
        self.items, self.targets = self.__preprocess()

        self.handle_masks = np.zeros_like(self.targets, dtype=np.bool)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index], self.noise_masks[index]

    def __preprocess(self):
        neg_items, neg_targets = self.__negative_sampling()
        items = np.concatenate((self.pos_items, neg_items), axis=0)
        targets = np.concatenate((self.pos_targets, neg_targets), axis=0)
        print("num of pos:", np.sum(targets), "neg:", len(targets) - np.sum(targets))
        self.noise_masks = np.concatenate((self.noise_masks, np.zeros_like(neg_targets, dtype=np.bool)), axis=0)
        return items, targets

    def __negative_sampling(self):
        num = self.num_neg * len(self.pos_targets)
        neg_users = np.random.randint(0, self.num_users, size=2 * num)
        neg_items = np.random.randint(0, self.num_items, size=2 * num)
        neg_features = np.concatenate((neg_users.reshape(-1, 1), neg_items.reshape(-1, 1)), axis=1)
        neg_features = np.array(list(set(map(tuple, neg_features)) - set(map(tuple, self.pos_items))))
        neg_features = neg_features[:min(num, len(neg_features))]
        neg_targets = np.zeros(len(neg_features), dtype=np.float32)
        return neg_features, neg_targets

    def clean_test(self, test_indices):
        test_indices = np.array(test_indices)
        print("********** test data **********")
        print(f"pre num of pos in test: {np.sum(self.targets[test_indices])} "
              f"tp: {np.sum(self.targets[test_indices[~self.noise_masks[test_indices]]])} "
              f"fp: {np.sum(self.targets[test_indices[self.noise_masks[test_indices]]])}")
        self.targets[test_indices[self.noise_masks[test_indices]]] = 0
        self.noise_masks[test_indices] = False
        print(f"post num of pos in test: {np.sum(self.targets[test_indices])} "
              f"tp: {np.sum(self.targets[test_indices[~self.noise_masks[test_indices]]])} "
              f"fp: {np.sum(self.targets[test_indices[self.noise_masks[test_indices]]])}")

    def handle_data(self, handle_indices):
        self.handle_masks[handle_indices] = True


class YelpDataset(CustomDataset):
    pass


class AmazonBookDataset(CustomDataset):
    pass


class AdressaDataset(CustomDataset):
    pass


class BookCrossingDataset(CustomDataset):
    pass


class JesterDataset(CustomDataset):
    pass


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    # dataset = YelpDataset("../../examples/data/yelp/yelp.dat")
    # dataset = AmazonBookDataset("../../examples/data/amazon_book/amazon_book.dat")
    # dataset = AdressaDataset("../../examples/data/adressa/adressa.dat")
    dataset = BookCrossingDataset("../../examples/book_crossing/book_crossing.dat")
    print(dataset.num_users)
    print(dataset.num_items)
