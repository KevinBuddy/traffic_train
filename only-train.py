from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
def addfiletolist(rootdir,img,val,id):
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            img1 = Image.open(path)  
            img1=img1.resize((32,32))
            arr1=np.array(img1)
            img.append(arr1)
            #print(arr1.shape)
            #print(path)
            #print(np.array(img).shape)
            val.append(id)
X_all=[]
y_all=[]
rootdir = 'image_red'
addfiletolist(rootdir,X_all,y_all,0)
rootdir = 'image_green'
addfiletolist(rootdir,X_all,y_all,1)
rootdir = 'image_yellow'
addfiletolist(rootdir,X_all,y_all,2)
rootdir = 'image_other'
addfiletolist(rootdir,X_all,y_all,3)

import numpy as np
import pandas
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
X_all=np.array(X_all)
y_all=np.array(y_all)

n_train = len(X_all)

# TODO: Number of validation examples
#n_validation = len(X_valid)

# TODO: Number of testing examples.
#n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_all.shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_all)+1

print("Number of training examples =", n_train)
#print("Number of validation examples =", n_validation)
#print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
from PIL import Image, ImageFilter
from PIL import Image, ImageFilter

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)
X_train, y_train = shuffle(X_all, y_all)
X_valid=X_train[n_train-200:n_train]
y_valid=y_train[n_train-200:n_train]
X_train=X_train[0:n_train-200]
y_train=y_train[0:n_train-200]
n_train=n_train-200

import tensorflow as tf
from tensorflow.contrib.layers import flatten
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w=tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6),mean=mu,stddev=sigma))
    conv1_b=tf.Variable(tf.zeros(6))
    conv1=tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding='VALID')+conv1_b
    # TODO: Activation.
    conv1=tf.nn.relu(conv1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w=tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),mean=mu,stddev=sigma))
    conv2_b=tf.Variable(tf.zeros(16))
    conv2=tf.nn.conv2d(conv1,conv2_w,strides=[1,1,1,1],padding='VALID')+conv2_b
    
    # TODO: Activation.
    conv2=tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fe0=flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w=tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma))
    fc1_b=tf.Variable(tf.zeros(120))
    fc1=tf.matmul(fe0,fc1_w)+fc1_b
    
    # TODO: Activation.
    fc1=tf.nn.relu(fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w=tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma))
    fc2_b=tf.Variable(tf.zeros(84))
    fc2=tf.matmul(fc1,fc2_w)+fc2_b
   
    # TODO: Activation.
    fc2=tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w=tf.Variable(tf.truncated_normal(shape=(84,43),mean=mu,stddev=sigma))
    fc3_b=tf.Variable(tf.zeros(43))
    logits_temp=tf.matmul(fc2,fc3_w)+fc3_b
    
    return logits_temp
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
topk = tf.nn.softmax(logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def evaluate_index(X_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    out = sess.run(tf.argmax(logits, 1), feed_dict={x: X_data})
    return out

def evaluate_top(X_data,k):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    out = sess.run(tf.nn.top_k(topk,k), feed_dict={x: X_data})
    #out = sess.run(tf.reduce_mean(logits), feed_dict={x: X_data})
    return out
### Define your architecture here.
### Feel free to use as many code cells as needed.
EPOCHS = 100
BATCH_SIZE = 10

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")

