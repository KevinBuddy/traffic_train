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

print("Number of training examples =", n_train)
X_train = X_all

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
rate = 0.001

logits = LeNet(x)

def evaluate_index(X_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    out = sess.run(tf.argmax(logits, 1), feed_dict={x: X_data})
    return out

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./lenet')
    validation_accuracy = evaluate_index(X_train)
    print (validation_accuracy)