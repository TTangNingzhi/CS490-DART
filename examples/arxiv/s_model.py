import torch
import torch.nn as nn

# Step 1. Baseline model
y_train_noise = torch.tensor([1, 0, 1, 0, 1, 0])
y_baseline_output = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
y_baseline_pred = torch.tensor([0, 1, 1, 0, 1, 0])

baseline_confusion = torch.zeros((2, 2))  # (2, 2) for binary classification
for n, p in zip(y_train_noise, y_baseline_pred):
    baseline_confusion[p, n] += 1.
print(baseline_confusion)

# Step 2. S-model
channel_weights = baseline_confusion.clone()
channel_weights /= channel_weights.sum(axis=1, keepdims=True)
# perm_bias_weights[prediction, noisy_label] = log(P(noisy_label|prediction))
channel_weights = torch.log(channel_weights + 1e-8)
print(channel_weights)
channel_layer = nn.Linear(2, 2, bias=False)
channel_layer.weight.data = channel_weights
channel_output = torch.softmax(channel_layer(y_baseline_output), dim=1)
print(channel_output)

# Step 3. Loss
beta = 0.8
loss = nn.CrossEntropyLoss()(channel_output, y_train_noise) * beta + \
       nn.CrossEntropyLoss()(y_baseline_output, y_train_noise) * (1 - beta)
