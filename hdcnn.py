#Setup and Imports
#Import Packages

import keras as kr
import numpy as np
import tensorflow as tf

from keras.datasets import cifar100

from sklearn.model_selection import train_test_split

from random import randint
import time
import os

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # Don't pre-allocate memory; allocate as-needed
config.log_device_placement = True  # to log device placement (on which device the operation ran)
#config.gpu_options.per_process_gpu_memory_fraction = 0.3 # Only allow a total fraction the GPU memory to be allocated
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

if not os.path.exists('data/models/'):
    os.mkdir('data/models')

# Define Global Variables
# The number of coarse categories
coarse_categories = 20

# The number of fine categories
fine_categories = 100

# Import and Preprocess Dataset
# Import Cifar100 Data Set
(X, y_c), (x_test, y_c_test) = cifar100.load_data(label_mode='coarse')
(X, y), (x_test, y_test) = cifar100.load_data(label_mode='fine')


# Fine-To-Coarse Mapping
# (Ideally, this would be done through spectral clustering as opposed to hard-coding)
fine2coarse = np.zeros((fine_categories,coarse_categories))
for i in range(coarse_categories):
    index = np.where(y_c_test[:,0] == i)[0]
    fine_cat = np.unique([y_test[j,0] for j in index])
    for j in fine_cat:
        fine2coarse[j,i] = 1

y_c = 0; # Clear y_c in interest of saving mem
y_c_test=0;

################################################################################
#    Title: One Hot Encoding
################################################################################
#    Description: 
#        This function extends a matrix to one-hot encoding
#    
#    Parameters:
#        y    Array of label values
# 
#    Returns:
#        y_new    One hot encoded array of labels
################################################################################
def one_hot(y):
    n_values = np.max(y) + 1
    y_new = np.eye(n_values)[y[:,0]]
    return y_new

y=one_hot(y)
y_test=one_hot(y_test)
print(np.shape(y))

################################################################################
#    Title: ZCA
################################################################################
#    Description: 
#        This function applies ZCA Whitening to the image set
#    
#    Parameters:
#        x_1           Array of MxNxC images to compute the ZCA Whitening
#        x_2           Array of MxNxC images to apply the ZCA transform
#        num_batch    Number of batches to do the computation
# 
#    Returns:
#        An array of MxNxC zca whitened images
################################################################################
def zca(x_1, x_2, epsilon=1e-5):
        
    with tf.name_scope('ZCA'):
        
        x1 = tf.placeholder(tf.float64, shape=np.shape(x_1), name='placeholder_x1')
        x2 = tf.placeholder(tf.float64, shape=np.shape(x_2), name='placeholder_x2')
        
        flatx = tf.cast(tf.reshape(x1, (-1, np.prod(x_1.shape[-3:])),name="reshape_flat"),tf.float64,name="flatx")
        sigma = tf.tensordot(tf.transpose(flatx),flatx, 1,name="sigma") / tf.cast(tf.shape(flatx)[0],tf.float64) ### N-1 or N?
        s, u, v = tf.svd(sigma,name="svd")
        pc = tf.tensordot(tf.tensordot(u,tf.diag(1. / tf.sqrt(s+epsilon)),1,name="inner_dot"),tf.transpose(u),1, name="pc")
        
        net1 = tf.tensordot(flatx, pc,1,name="whiten1")
        net1 = tf.reshape(net1,np.shape(x_1), name="output1")
        
        flatx2 = tf.cast(tf.reshape(x2, (-1, np.prod(x_2.shape[-3:])),name="reshape_flat2"),tf.float64,name="flatx2")
        net2 = tf.tensordot(flatx2, pc,1,name="whiten2")
        net2 = tf.reshape(net2,np.shape(x_2), name="output2")
        
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_1,x_2 = sess.run([net1,net2], feed_dict={x1: x_1, x2: x_2})    
    return x_1,x_2

time1 = time.time()
X,x_test = zca(X,x_test)
time2 = time.time()
print('Time Elapsed - ZCA Whitening: '+str(time2-time1));


# Split Training set into Training and Validation sets
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=.1, random_state=0)
X = 0
y = 0


# Flip, pad and randomly crop each photo
################################################################################
#    Title: Preprocess Img
################################################################################
#    Description: 
#        This function pads images by 4 pixels, randomly crops them, then
#        randomly flips them
#    
#    Parameters:
#        x_1           Array of MxNxC images to compute the ZCA Whitening
#        x_2           Array of MxNxC images to apply the ZCA transform
#        num_batch    Number of batches to do the computation
# 
#    Returns:
#        An array of MxNxC zca whitened images
################################################################################
def preprocess_img(X,y):
        
    with tf.name_scope('Preproc'):
        
        images = tf.placeholder(tf.float64, shape=np.shape(X))
        labels = tf.placeholder(tf.float64, shape=np.shape(y))
        
        net = tf.map_fn(lambda img: tf.image.flip_left_right(img), images)
        net = tf.map_fn(lambda img: tf.image.rot90(img), net)
        net = tf.image.resize_image_with_crop_or_pad(net,40,40)
        net = tf.map_fn(lambda img: tf.random_crop(img, [32,32,3]), net)

        net1 = tf.image.resize_image_with_crop_or_pad(images,40,40)
        net1 = tf.map_fn(lambda img: tf.random_crop(img, [32,32,3]), net1)
        
        net = tf.concat([net, net1],0)
        net = tf.random_shuffle(net, seed=0)
        net_labels = tf.concat([labels, labels],0)
        net_labels = tf.random_shuffle(net_labels,seed=0)
        
        net = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), net)
        
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_t,y_t = sess.run([net,net_labels], feed_dict={images: X, labels: y})    
    return x_t,y_t

time1 = time.time()
x_train,y_train = preprocess_img(x_train,y_train)
time2 = time.time()
print('Time Elapsed - Img Preprocessing: '+str(time2-time1));

# Single Classifier Training
# Constructing CNN
from keras import optimizers
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Model

in_layer = Input(shape=(32, 32, 3), dtype='float32', name='main_input')

net = Conv2D(384, 3, strides=1, padding='same', activation='elu')(in_layer)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(384, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(384, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(640, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.2)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(640, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(768, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.3)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(768, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(896, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.4)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(896, 3, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1024, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.5)(net)
net = MaxPooling2D((2, 2), padding='valid')(net)

net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(net)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
net = Dense(100, activation='softmax')(net)


# Compile Model
model = Model(inputs=in_layer,outputs=net)
sgd_coarse = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy'])
#model.load_weights('data/models/model_coarse'+str(30))

# Train Model
tbCallBack = kr.callbacks.TensorBoard(log_dir='./data/graph/elu_drop/', histogram_freq=0, write_graph=True, write_images=True)
batch = 64
index= 0
step = 5
stop = 30

while index < stop:
    model.fit(x_train, y_train, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val), callbacks=[tbCallBack])
    index += step
    model.save_weights('data/models/model_coarse'+str(index))
save_index = index

# Load Most Recent Model
sgd_fine = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
for i in range(len(model.layers)):
    model.layers[i].trainable=False


# Fine-Tuning for Coarse Classifier
y_train_c = np.dot(y_train,fine2coarse)
y_val_c = np.dot(y_val,fine2coarse)

net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
net = Dropout(.6)(net)
net = MaxPooling2D((2, 2), padding='same')(net)

net = Flatten()(net)
net = Dense(1152, activation='elu')(net)
out_coarse = Dense(20, activation='softmax')(net)

model_c = Model(inputs=in_layer,outputs=out_coarse)
model_c.compile(optimizer= sgd_coarse, loss='categorical_crossentropy', metrics=['accuracy'])

for i in range(len(model_c.layers)-1):
    model_c.layers[i].set_weights(model.layers[i].get_weights())

index = 30
step = 10
stop = 40

while index < stop:
    model_c.fit(x_train, y_train_c, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val_c), callbacks=[tbCallBack])
    index += step

model_c.compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy'])
stop = 50

while index < stop:
    model_c.fit(x_train, y_train_c, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_val, y_val_c), callbacks=[tbCallBack])
    index += step

# Fine-Tuning for Fine Classifiers
# Construct Fine Classifiers
def fine_model():
    net = Conv2D(1024, 1, strides=1, padding='same', activation='elu')(model.layers[-8].output)
    net = Conv2D(1152, 2, strides=1, padding='same', activation='elu')(net)
    net = Dropout(.6)(net)
    net = MaxPooling2D((2, 2), padding='same')(net)

    net = Flatten()(net)
    net = Dense(1152, activation='elu')(net)
    out_fine = Dense(100, activation='softmax')(net)
    model_fine = Model(inputs=in_layer,outputs=out_fine)
    model_fine.compile(optimizer= sgd_coarse,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    for i in range(len(model_fine.layers)-1):
        model_fine.layers[i].set_weights(model.layers[i].get_weights())
    return model_fine

fine_models = {'models' : [{} for i in range(coarse_categories)], 'yhf' : [{} for i in range(coarse_categories)]}
for i in range(coarse_categories):
    model_i = fine_model()
    fine_models['models'][i] = model_i

# Train Fine Classifiers on Respective Data
def get_error(y,yh):
    # Threshold 
    yht = np.zeros(np.shape(yh))
    yht[np.arange(len(yh)), yh.argmax(1)] = 1
    # Evaluate Error
    error = np.count_nonzero(np.count_nonzero(y-yht,1))/len(y)
    return error

for i in range(coarse_categories):
    index= 0
    step = 5
    stop = 5
    
    # Get all training data for the coarse category
    ix = np.where([(y_train[:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
    x_tix = x_train[ix]
    y_tix = y_train[ix]
    
    # Get all validation data for the coarse category
    ix_v = np.where([(y_val[:,j]==1) for j in [k for k, e in enumerate(fine2coarse[:,i]) if e != 0]])[1]
    x_vix = x_val[ix_v]
    y_vix = y_val[ix_v]
    
    while index < stop:
        fine_models['models'][i].fit(x_tix, y_tix, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_vix, y_vix))
        index += step
    
    fine_models['models'][i].compile(optimizer=sgd_fine, loss='categorical_crossentropy', metrics=['accuracy'])
    stop = 10

    while index < stop:
        fine_models['models'][i].fit(x_tix, y_tix, batch_size=batch, initial_epoch=index, epochs=index+step, validation_data=(x_vix, y_vix))
        index += step
        
    yh_f = fine_models['models'][i].predict(x_val[ix_v], batch_size=batch)
    print('Fine Classifier '+str(i)+' Error: '+str(get_error(y_val[ix_v],yh_f)))


# Probabilistic Averaging

def eval_hdcnn(X, y):
    yh = np.zeros(np.shape(y))
    
    yh_s = model.predict(X, batch_size=batch)
    
    print('Single Classifier Error: '+str(get_error(y,yh_s)))
    
    yh_c = model_c.predict(X, batch_size=batch)
    y_c = np.dot(y,fine2coarse)
    
    print('Coarse Classifier Error: '+str(get_error(y_c,yh_c)))

    for i in range(coarse_categories):
        if i%5 == 0:
            print("Evaluating Fine Classifier: ", str(i))
        #fine_models['yhf'][i] = fine_models['models'][i].predict(X, batch_size=batch)
        yh += np.multiply(yh_c[:,i].reshape((len(y)),1), fine_models['yhf'][i])
    
    print('Overall Error: '+str(get_error(y,yh)))
    return yh

yh = eval_hdcnn(x_val,y_val)
print("yh\n", yh)

