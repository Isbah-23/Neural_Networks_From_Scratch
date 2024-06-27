import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.utils

class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(4,16)
        self.fc2 = torch.nn.Linear(16,3)

    def forward(self, inputs):
        layer_output = torch.relu(self.fc1(inputs))
        layer_output = self.fc2(layer_output)
        return layer_output
    
model = ANN()
loss_function = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.SGD(model.parameters(), lr=0.001)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)
y_train = torch.tensor(y_train).to(torch.long)
X_train = torch.tensor(X_train).to(torch.float32)

dataset = torch.utils.data.TensorDataset(X_train,y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 42, shuffle=True)
epochs = 5001
for epoch in range(epochs):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

    if not epoch % 500:
        print(f"In Epoch: {epoch} Loss is {loss.item():.4f}")

X_test = torch.tensor(X_test).to(torch.float32)
y_test = torch.tensor(y_test).to(torch.long)

dataset = torch.utils.data.TensorDataset(X_test, y_test)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=42, shuffle=False)

model.eval()
correct = total = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        output = model.forward(inputs)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct/total * 100
print(f"Test Accuracy: {accuracy:.4f}%")