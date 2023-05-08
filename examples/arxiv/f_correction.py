import torch
import torch.nn.functional as F

# Define the noisy labels y_hat (n x c) and the true labels y (n)
y_hat = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
y = torch.tensor([0, 1, 2])

# Compute the empirical confusion matrix T_emp (c x c)
T_emp = torch.zeros((3, 3))
for i in range(3):
    for j in range(3):
        T_emp[i][j] = ((y == i) & (y_hat.argmax(dim=1) == j)).sum().item() / (y == i).sum().item()

print("Empirical confusion matrix:\n", T_emp)

# Compute the smoothed confusion matrix T_smoothed (c x c)
alpha = 1e-6
T_smoothed = (1 - alpha) * T_emp + alpha / y_hat.shape[1]

print("Smoothed confusion matrix:\n", T_smoothed)

# Define the confusion matrix T (c x c) and its inverse T_inv
T = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
# T = T_smoothed
T_inv = torch.pinverse(T)

# Compute the noisy predictions p_hat (n x c) and the true predictions p (n x c)
p_hat = y_hat @ T
p = F.one_hot(y, num_classes=3).float()

# Compute the forward-corrected loss L_fc using negative log-likelihood
L_fc = F.nll_loss(torch.log(p_hat), y)

# Compute the cross-entropy loss L_ce using negative log-likelihood
L_ce = F.nll_loss(torch.log(y_hat), y)

print("Forward-corrected loss:", L_fc.item())
print("Cross-entropy loss:", L_ce.item())
