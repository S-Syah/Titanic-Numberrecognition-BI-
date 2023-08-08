import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
m=keras.datasets.mnist.load_data()
(x_train,y_train),(x_test,y_test)=m
x_train
x_train.shape
x_test.shape
y_train
import matplotlib.pyplot as plt
plt.imshow(x_train[56])
x_test=x_test/255
x_train=x_train/255
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
h=model.fit(x_train, y_train, epochs=10, validation_split=0.2)
p=model.predict(x_test)
y_pred=p.argmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
accuracy = model.evaluate(x_test, y_test)[1]
print("Test Accuracy:", accuracy)
plt.matshow(x_test[1])
np.argmax(y_pred[1])
y_pred=model.predict(x_test)
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
y_predictedlabels=[np.argmax(i) for i in y_pred]
c=tf.math.confusion_matrix(labels=y_test,predictions=y_predictedlabels)
import seaborn as sns
pl.figure(figsize=(10,7))
sns.heatmap(c,annot=True,fmt='d')
pl.xlabel("predicted")
pl.ylabel("truth")
