import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class Layer():
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
    def backward(self, deltas):
        self.delta_weights = np.dot(self.inputs.T, deltas)
        self.delta_inputs = np.dot(deltas, self.weights.T)
        self.delta_biases = np.sum(deltas, axis = 0, keepdims=True)

class ReLU():
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, deltas):
        self.delta_inputs = deltas.copy()
        self.delta_inputs[self.inputs <= 0] = 0
    
class Softmax():
    def forward(self, inputs):
        self.inputs = inputs
        exp_vals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_vals/np.sum(exp_vals, axis=1, keepdims=True)

class Loss():
    def calculate(self, output, y_true):
        sample_loss = self.forward(output, y_true)
        batch_loss = np.mean(sample_loss)
        return batch_loss
    
class Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        num_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)
        accuracy = y_pred_clipped[range(num_samples),y_true]
        negative_log = -np.log(accuracy)
        return negative_log
    
class Activation_Softmax_Loss_Categorical_CrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Categorical_Cross_Entropy()
    def forward(self, inputs, y_true):
        self.inputs = inputs
        self.activation.forward(self.inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, deltas, y_true):
        num_samples = len(deltas)
        self.delta_inputs = deltas.copy()
        self.delta_inputs[range(num_samples), y_true] -= 1
        self.delta_inputs = self.delta_inputs / num_samples

class Optimizer_SGD():
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        weight_updates = -self.learning_rate*layer.delta_weights
        bias_updates = -self.learning_rate*layer.delta_biases
        layer.weights += weight_updates
        # print("Biases shape: ",layer.biases.shape)
        layer.biases += bias_updates
        

# Create dataset
iris_dataset = pd.read_csv('Iris.csv')
iris_dataset = iris_dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]
X = iris_dataset.drop(columns=['Species'])
y = iris_dataset['Species']

# over here i shall convert my labels to sparse vector to prevent my custom implementation from dying
label_encoder = LabelEncoder()
y = np.array(label_encoder.fit_transform(y)) # perfecto

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

d1 = Layer(4,32)
a1 = ReLU()
d2 = Layer(32, 3)
a2 = Activation_Softmax_Loss_Categorical_CrossEntropy()
o = Optimizer_SGD(learning_rate=0.0001)

curr_acc = 0
curr_params = []

for epoch in range(10000):
    d1.forward(X_train)
    a1.forward(d1.output)
    d2.forward(a1.output)
    loss = a2.forward(d2.output, y_train)

    predictions = np.argmax(a2.output, axis=1)
    accuracy = np.mean(predictions == y_train)
    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}')
        # print("D1 weights: ", d1.weights)
        # print("D2 weights: ", d2.weights)
        if accuracy > curr_acc:
            curr_acc = accuracy
            curr_params = [d1.weights, d1.biases, d2.weights, d2.biases] # save best params

    a2.backward(a2.output, y_train)
    d2.backward(a2.delta_inputs)
    a1.backward(d2.delta_inputs)
    d1.backward(a1.delta_inputs)
    o.update_params(d1)
    o.update_params(d2)

# restore best params
d1.weights, d1.biases, d2.weights, d2.biases = curr_params[0], curr_params[1], curr_params[2], curr_params[3]

d1.forward(X_test)
a1.forward(d1.output)
d2.forward(a1.output)
loss = a2.forward(d2.output, y_test)

predictions = np.argmax(a2.output, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'Testing; acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}')
# plt.figure(figsize=(10,7))
# cm = confusion_matrix(predictions, y_test)
# sns.heatmap(cm, annot=True, cmap="Blues")
# plt.title('Confusion Matrix')
# plt.show()
# print(classification_report(predictions,y_test))