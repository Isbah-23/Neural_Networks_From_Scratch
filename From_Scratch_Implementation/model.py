from components import Layer, ReLU, Activation_Softmax_Loss_Categorical_CrossEntropy, Optimizer_SGD
from matrix import Matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

with open("report.txt","w") as outfile:
    outfile.write('Takes eons to run without numpy so i\'ll record my results here\n')

# Create dataset
iris_dataset = pd.read_csv('Iris.csv')
iris_dataset = iris_dataset[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]
X = iris_dataset.drop(columns=['Species'])
y = iris_dataset['Species']

# over here i shall convert my labels to sparse vector to prevent my custom implementation from dying
label_encoder = LabelEncoder()
y = list(label_encoder.fit_transform(y)) # perfecto

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

d1 = Layer(4,32)
a1 = ReLU()
d2 = Layer(32, 3)
a2 = Activation_Softmax_Loss_Categorical_CrossEntropy()
o = Optimizer_SGD(learning_rate=0.0001)

curr_acc = 0
curr_params = []
X_train = Matrix(X_train.values.tolist())
X_test = Matrix(X_test.values.tolist())

for epoch in range(10000):
    d1.forward(X_train)
    a1.forward(d1.output)
    d2.forward(a1.output)
    loss = a2.forward(d2.output, y_train)

    predictions = a2.output.argmax()
    accuracy = (predictions.acc(y_train)).mean()

    if epoch % 1000 == 0:
        with open("report.txt","a") as outfile:
            outfile.write(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}\n')
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}')


    a2.backward(a2.output, y_train)
    d2.backward(a2.delta_inputs)
    a1.backward(d2.delta_inputs)
    d1.backward(a1.delta_inputs)

    o.update_params(d1)
    o.update_params(d2)

d1.forward(X_test)
a1.forward(d1.output)
d2.forward(a1.output)
loss = a2.forward(d2.output, y_test)

predictions = a2.output.argmax()
accuracy = (predictions.acc(y_test)).mean()
with open("report.txt","a") as outfile:
    outfile.write(f'Testing; acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}\n')
print(f'Testing; acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {o.learning_rate}')