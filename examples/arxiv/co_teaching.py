import torch
import torch.nn.functional as F
import numpy as np


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.data)

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.data)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(ind_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember


if __name__ == '__main__':
    y_1 = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
    y_2 = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.2, 0.8], [0.8, 0.2]])
    t = torch.tensor([0, 1, 1, 0])
    forget_rate = 0.05
    loss_1, loss_2 = loss_coteaching(y_1, y_2, t, forget_rate)
    print(loss_1, loss_2)


