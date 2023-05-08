import config

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens100KDataset, MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork


def get_dataset(name, path):
    if name == 'movielens100K':
        return MovieLens100KDataset(path)
    elif name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path, cache_path='./cache/criteo')
    elif name == 'avazu':
        return AvazuDataset(path, cache_path='./cache/avazu')
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400),
            dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100, epoch_i=0, method="none"):
    assert method in ["none", "discard", "relabel", "reweight"]
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    num_hit, num_noise, num_relabel = 0, 0, 0
    total_pos_loss, total_neg_loss = 0, 0
    pos_num, neg_num = 0, 0

    none_method = epoch_i < 1 or method == "none"  # movielens 10

    for i, (fields, target, mask) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        with torch.no_grad():
            loss_val = F.binary_cross_entropy(y, target.float(), reduction='none')
            total_pos_loss += torch.sum(loss_val[target == 1]).item()
            total_neg_loss += torch.sum(loss_val[target == 0]).item()
            pos_num += torch.sum(target == 1).item()
            neg_num += torch.sum(target == 0).item()
        if none_method:
            loss = criterion(y, target.float())
        else:
            negative_rate = 0.01
            negative_num = int(negative_rate * len(target))
            negative_index = torch.argsort(loss_val, descending=True)[0:negative_num]

            target_relabel = target.clone()
            target_relabel[negative_index] = 0
            relabel_mask = target != target_relabel
            num_hit += torch.sum(relabel_mask.cpu() & mask).item()
            num_noise += torch.sum(mask).item()
            num_relabel += torch.sum(relabel_mask).item()
            if method == 'relabel':
                loss = criterion(y, target_relabel.float())  # relabel
            elif method == 'discard':
                loss = criterion(y[~relabel_mask], target_relabel[~relabel_mask].float())  # discard
            else:
                loss = criterion(y, target.float())  # none
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    if not none_method and num_noise > 0:
        precision = num_hit / num_relabel
        recall = num_hit / num_noise
        print('precision: {:.4f}, recall: {:.4f}'.format(precision, recall))

    print('avg pos loss: {:.4f}, avg neg loss: {:.4f}'.format(total_pos_loss / pos_num, total_neg_loss / neg_num))


def train_1(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target, mask) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target, mask in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def handle(model, train_data_loader, device):
    model.eval()
    targets, predicts, losses = list(), list(), list()
    with torch.no_grad():
        for fields, target, mask in tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = F.binary_cross_entropy(y, target.float(), reduction='none')
            losses.extend(loss.tolist())
    threshold = np.percentile(losses, 99)
    dataset = train_data_loader.dataset
    train_indices = np.array(dataset.indices, dtype=int)
    handle_indices = train_indices[np.where(np.array(losses) > threshold)[0].astype(int)].tolist()
    return handle_indices


def eval_loss(model, data_loader, device):
    model.eval()
    losses = list()
    with torch.no_grad():
        for fields, target, mask in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = F.binary_cross_entropy(y, target.float(), reduction='none')
            losses.extend(loss.tolist())
    return losses


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         noise_rate,
         method,
         mode):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    np.save(f'logs/train_indices.npy', np.array(train_dataset.indices, dtype=int))
    dataset.noisify(train_dataset.indices, noise_rate)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    if mode == 'static':
        for epoch_i in range(1):
            train_1(model, optimizer, train_data_loader, criterion, device)
            auc = test(model, valid_data_loader, device)
            print('*** pre-train - epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
        handle_indices = handle(model, train_data_loader, device)
        print(f'*** handle - num_handle: {len(handle_indices)}')
        dataset.relabel(handle_indices)
        model = get_model(model_name, dataset).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
        for epoch_i in range(epoch):
            train_1(model, optimizer, train_data_loader, criterion, device)
            auc = test(model, valid_data_loader, device)
            print('*** post-train - epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
    else:
        for epoch_i in range(epoch):
            if epoch_i == 1:
                np.save(f'logs/targets.npy', np.array(dataset.targets, dtype=bool))
                np.save(f'logs/noise_mask.npy', np.array(dataset.noise_masks, dtype=bool))
            losses = eval_loss(model, train_data_loader, device)
            np.save(f'logs/losses_{epoch_i}.npy', np.array(losses, dtype=float))
            train(model, optimizer, train_data_loader, criterion, device,
                  epoch_i=epoch_i, method=method)
            auc = test(model, valid_data_loader, device)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break
        losses = eval_loss(model, train_data_loader, device)
        np.save(f'logs/losses_{epoch}.npy', np.array(losses, dtype=float))
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    import argparse
    from seed import set_all_seeds

    set_all_seeds(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, ml-100k/u.data, or ml-1m/ratings.dat',
                        default='criteo/train.txt')
    parser.add_argument('--model_name', default='deepfm')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='none', help='none, relabel, or discard')
    parser.add_argument('--mode', type=str, default='static', help='static or dynamic')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.noise_rate,
         args.method,
         args.mode)
