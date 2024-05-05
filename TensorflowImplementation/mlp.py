from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Iris.csv')
df = df.drop(columns=['Id'])
print(df)
X = df.drop(columns=['Species'])
y = df['Species']
label_encoder = LabelEncoder()
y_sparse = label_encoder.fit_transform(df['Species'])
print(y_sparse)
X_train, X_test, y_train, y_test = train_test_split(X, y_sparse, test_size=0.2, random_state=42)
model = Sequential([
    Dense(16,activation='relu'),
    Dense(3,activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)
print("Test Results")
model.evaluate(X_test, y_test)