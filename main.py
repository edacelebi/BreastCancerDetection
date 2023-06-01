import os
import time
from matrix_curve import matrix_curve
import numpy as np
import cv2
from tqdm import tqdm
from model import model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import warnings as wr
wr.filterwarnings('ignore')

categories = ['benign', 'malignant', 'normal']
x = []
y = []
image_size = 32

for i in categories:
    folderPath = os.path.join('Dataset/', i)
    for j in tqdm(os.listdir(folderPath)):
        image = cv2.imread(os.path.join(folderPath, j),0)
        image = cv2.resize(image, (image_size, image_size))
        x.append(image)
        y.append(i)


x = np.array(x)
y = np.array(y)

x =x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

print(x.shape)
x, y = shuffle(x, y, random_state=42)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=42)

y_train_new = []
y_test_new = []

for i in Y_train:
    y_train_new.append(categories.index(i))
Y_train = to_categorical(y_train_new)

for i in Y_test:
    y_test_new.append(categories.index(i))
Y_test = to_categorical(y_test_new)


X_train_scaled = X_train.astype('float32')
X_test_scaled = X_test.astype('float32')

X_train_scaled /= 255
X_test_scaled /= 255

model = model().lenet5()

start_time = time.time()

history = model.fit(x=X_train_scaled, y=Y_train,
                   batch_size=32,
                   epochs=100,
                   validation_split = 0.20,
                   verbose=1)
test_loss,test_acc = model.evaluate(X_test_scaled, Y_test)
print("ACCURACY:", test_acc)

end_time = time.time()
print("Taken Time:", (end_time - start_time))

test_predictions = model.predict(X_test_scaled)
preds = np.argmax(test_predictions, axis=1)
actual_label = np.argmax(Y_test, axis=1)
print(classification_report(actual_label, preds))

cnf = confusion_matrix(actual_label, preds)

matrix_curve().plot_accuracy_loss(history)
matrix_curve().plot_cofusion_matrix(cnf)
