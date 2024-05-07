from matrix import Matrix # custom numpy
import random

# first lets make our layers - of neurons
class Layer():  # num weights = num_neurons * num_inputs
    def __init__(self, n_inputs, n_neurons):
        self.weights = Matrix([[random.gauss(0, 1) for _ in range(n_neurons)] for _ in range(n_inputs)]) # numpy thing without numpy in one line cause its cleaner
        self.biases = Matrix([[0] * n_neurons for _ in range(1)])
    def forward(self, inputs):
        self.inputs = Matrix(inputs) # need to remember for later in back prop
        self.output = self.inputs.dot(self.weights) + self.biases # + [1,2,3]
    def backward(self, deltas):
        deltas = Matrix(deltas)
        self.delta_weights = self.inputs.T().dot(deltas)
        self.delta_inputs = deltas.dot(self.weights.T())
        self.delta_biases = deltas.col_sum_arr()

class ReLU(): # hidden activations favourite child
    def forward(self, inputs):
        self.inputs = inputs
        self.output = Matrix([[max(0,elem) for elem in row] for row in inputs.matrix])
    def backward(self, deltas):
        self.delta_inputs = Matrix(deltas)
        for i in range(len(self.delta_inputs.matrix)):
            for j in range(len(self.delta_inputs.matrix[i])):
                if self.inputs.matrix[i][j] == 0:
                    self.delta_inputs.matrix[i][j] = 0
    
class Softmax(): # for classifiers
    def forward(self, inputs):
        self.inputs = Matrix(inputs)
        exp_vals = (inputs - inputs.max()).exp()
        self.output = exp_vals/exp_vals.row_sum_arr()

class Loss():
    def calculate(self, output, y_true):
        sample_loss = self.forward(output, y_true)
        batch_loss = sample_loss.mean()
        return batch_loss
    
class Categorical_Cross_Entropy(Loss):# the classifiers (softmax) loss function
    def forward(self, y_pred, y_true):
        if isinstance(y_pred, Matrix):
            num_samples = len(y_pred.matrix)
        else:
            num_samples = len(y_pred)
            y_pred = Matrix(y_pred)
        y_pred_clipped = y_pred.clip(1e-7,1-1e-7)
        accuracy = []
        for i in range(num_samples):
            accuracy.append(y_pred_clipped.matrix[i][y_true[i]]) # in row the value of the neuron which is supposed to be the correct class
        accuracy = Matrix(accuracy)
        negative_log = -accuracy.log()
        return negative_log
    
class Activation_Softmax_Loss_Categorical_CrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = Categorical_Cross_Entropy()
    def forward(self, inputs, y_true):
        self.inputs = Matrix(inputs)
        self.activation.forward(self.inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, deltas, y_true):
        deltas = Matrix(deltas)
        num_samples = len(deltas.matrix)
        self.delta_inputs = Matrix(deltas)
        for i in range(num_samples):
            self.delta_inputs.matrix[i][y_true[i]] -= 1
        self.delta_inputs = self.delta_inputs / num_samples

class Optimizer_SGD(): # plain ol' sgd nothing fancy here
    def __init__(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        weight_updates = layer.delta_weights*(-self.learning_rate)
        bias_updates = layer.delta_biases*(-self.learning_rate)
        layer.weights += weight_updates
        layer.biases += bias_updates