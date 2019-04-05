import tensorflow as tf
from tensorflow import keras
import mnist_reader
import numpy as np
import matplotlib.pyplot as plt
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
#print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)'''
#plt.show()    #显示训练集中的第一个
#plt.figure(figsize=(10,10))

X_train=X_train/255.0
X_test=X_test/255.0
'''for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])

plt.show()#显示25张图片'''


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5)
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(X_test)

predictions[0]
print(np.argmax(predictions[0]))
print(y_test[0])

def plot_image(i,predictions_array,true_label,img):
    predictions_array,true_label,img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img,cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color='red'

    plt.xlabel("{} {:2.0f}%({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
               color=color)

def plot_value_array(i,predictions_array,true_label):
    predictions_array,true_label = predictions_array[i],true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_array,color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

'''i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,y_test,X_test)
plt.subplot(1,2,2)
plot_value_array(i,predictions,y_test)


i=12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions,y_test,X_test)
plt.subplot(1,2,2)
plot_value_array(i,predictions,y_test)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols,2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows,2*num_cols,2*i+1)
    plot_image(i,predictions,y_test,X_test)
    plt.subplot(num_rows,2*num_cols,2*i+2)
    plot_value_array(i,predictions,y_test)'''
img = X_test[0]

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)
plot_value_array(0,predictions_single,y_test)
_ = plt.xticks(range(10),class_names,rotation=45)
np.argmax(predictions_single[0])
plt.show()

