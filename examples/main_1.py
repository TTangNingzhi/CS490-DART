import config

import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens_1 import MovieLens100KDataset, MovieLens1MDataset, MovieLens20MDataset
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

tp_loss_iter, fp_loss_iter, neg_loss_iter, hit_noise_handle_iter = [], [], [], []


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
    assert method in ["none", "discard", "relabel", "reweight", "bootstrap_hard", "bootstrap_soft", 'lcd', 'lcd-re']
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    num_hit, num_noise, num_handle = 0, 0, 0

    global tp_loss_iter, fp_loss_iter, neg_loss_iter, hit_noise_handle_iter
    for i, (fields, target, mask) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        with torch.no_grad():
            num_iter = len(tp_loss_iter)
            loss_val = F.binary_cross_entropy(y, target.float(), reduction='none')
            tp_loss_iter.append(loss_val[~mask.to(device) & (target == 1)].mean().item())
            fp_loss_iter.append(loss_val[mask.to(device)].mean().item())
            neg_loss_iter.append(loss_val[target == 0].mean().item())
        if method == "none":
            loss = criterion(y, target.float())
        elif method in ["discard", "relabel"]:
            # handle_rate = min(0.05, 0.005 * epoch_i)  # MovieLens
            # handle_rate = 0 if epoch_i == 0 else 0.01  # Criteo/Avazu
            handle_rate = max(min(0.01, 0.01 / 17907 * num_iter), 0)  # Criteo/Avazu
            high_loss_num = int(handle_rate * len(target))
            high_loss_index = torch.argsort(loss_val, descending=True)[0:high_loss_num]
            handle_mask = torch.zeros_like(target, dtype=torch.bool)
            handle_mask[high_loss_index] = True
            handle_mask[target == 0] = False
            if method == "discard":
                loss = criterion(y[~handle_mask], target[~handle_mask].float())
            elif method == "relabel":
                target_relabel = target.clone()
                target_relabel[high_loss_index] = 0
                loss = criterion(y, target_relabel.float())
            # calculate num_hit, num_noise, num_handle
            iter_hit = torch.sum(handle_mask.cpu() & mask).item()
            iter_noise = torch.sum(mask).item()
            iter_handle = torch.sum(handle_mask).item()
            hit_noise_handle_iter.append([iter_hit, iter_noise, iter_handle])
            num_hit += iter_hit
            num_noise += iter_noise
            num_handle += iter_handle
        elif method == "reweight":
            alpha = 0.2
            t = target.float()
            loss = F.binary_cross_entropy(y, t, reduction='none')
            y_ = torch.sigmoid(y).detach()
            weight = torch.pow(y_, alpha) * t + torch.pow((1 - y_), alpha) * (1 - t)
            loss = torch.mean(loss * weight)
        elif method == "bootstrap_hard":
            alpha = 0.8
            cor_target = alpha * target.float() + (1 - alpha) * (y > 0.5).float()
            loss = criterion(y, cor_target)
        elif method == "bootstrap_soft":
            alpha = 0.95
            cor_target = alpha * target.float() + (1 - alpha) * y
            loss = criterion(y, cor_target.detach())
        elif method in ["lcd", "lcd-re"]:
            if epoch_i < 1:  # Criteo/Avazu 1 & MovieLens100K 10
                loss = criterion(y, target.float())
            else:
                weight = (loss_val - torch.min(loss_val)) / (torch.max(loss_val) - torch.min(loss_val) + 1e-8)
                # weight = torch.distributions.beta.Beta(2, 2).cdf(weight)
                if method == "lcd":
                    cor_target = weight * target.float() + (1 - weight) * y
                else:
                    cor_target = weight * y + (1 - weight) * target.float()
                loss = criterion(y, cor_target.detach())
        else:
            raise ValueError("unknown method: " + method)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

    if method in ['discard', 'relabel'] and num_noise > 0 and num_handle > 0:
        precision = num_hit / num_handle
        recall = num_hit / num_noise
        print('precision: {:.4f}, recall: {:.4f}'.format(precision, recall))


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
         method):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    dataset.noisify(train_dataset.indices, noise_rate)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device, epoch_i=epoch_i, method=method)
        auc = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuracy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')
    np.save(f'{save_dir}/{dataset_name}_{model_name}_{method}_tp_loss_iter.npy', np.array(tp_loss_iter))
    np.save(f'{save_dir}/{dataset_name}_{model_name}_{method}_fp_loss_iter.npy', np.array(fp_loss_iter))
    np.save(f'{save_dir}/{dataset_name}_{model_name}_{method}_neg_loss_iter.npy', np.array(neg_loss_iter))
    np.save(f'{save_dir}/{dataset_name}_{model_name}_{method}_hit_noise_handle_iter.npy',
            np.array(hit_noise_handle_iter))


if __name__ == '__main__':
    import argparse
    from config import set_all_seeds

    set_all_seeds(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo')
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, ml-100k/u.data, or ml-1m/ratings.dat',
                        default='criteo/train.txt')
    parser.add_argument('--model_name', default='fm')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='none', help='none, relabel, discard, reweight, lcd, lcd-re, '
                                                                   'bootstrap_soft, or bootstrap_hard')
    args = parser.parse_args()
    print(args)
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
         args.method)
