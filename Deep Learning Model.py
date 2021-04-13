import pandas as pd

df = pd.read_csv('creditcard.csv')

data = df.iloc[:, :-1]
targets = df.iloc[:,-1]

import numpy as np

rf_data = data

mean = np.mean(rf_data)
std = np.std(rf_data)

rf_data -= mean
rf_data /= std

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=5, random_state=0)
rf = clf.fit(rf_data, targets)

import matplotlib.pyplot as plt
features = data.columns
importance = rf.feature_importances_
indices = np.argsort(importance)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

from sklearn.model_selection import train_test_split
data_adjust = data[features[indices][-8:]]
train_data, test_data, train_targets, test_targets = train_test_split(data_adjust, targets, test_size=.3)

train_data, validation_data, train_targets, validation_targets = train_test_split(train_data, train_targets, test_size=.2)

import numpy as np

mean = np.mean(train_data)
std = np.std(train_data)


train_data -= mean
train_data /= std

validation_data -= mean
validation_data /= std

test_data -= mean
test_data /= std

from tensorflow import keras
from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(8, input_shape=(train_data.shape[1],), activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = keras.optimizers.SGD(lr=1e-7, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(patience=10)

history = model.fit(train_data,
                    train_targets,
                    epochs=100,
                    validation_data=(validation_data, validation_targets),
                    callbacks= [early_stopping])

import matplotlib.pyplot as plt
loss = history.history['accuracy']
val_loss = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training accuracy')
plt.plot(epochs, val_loss, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.evaluate(test_data, test_targets)