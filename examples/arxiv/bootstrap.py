import torch


# Define your model
class MyModel(torch.nn.Module):

    def init(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyModel()

# Define your loss function
criterion = torch.nn.CrossEntropyLoss()

# Define your optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train your model using bootstrapping
num_epochs = 10
alpha = 0.8
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):  # Forward pass
        outputs = model(inputs)
        # Compute the convex combination of training labels and the current model's predictions
        # Hard targets
        targets = (1 - alpha) * labels + alpha * outputs.argmax(dim=1)
        # # Soft targets
        # targets = (1 - alpha) * labels + alpha * outputs.max(dim=1)
        # Compute the loss
        loss = criterion(outputs, targets.long())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
