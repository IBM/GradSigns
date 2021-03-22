
##### This code works on Keras version 2.2.4 with Tensorflow nightly version 1.14.1-dev20190402

from __future__ import print_function
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar10
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import scipy.io as sio
import random,pickle
from os import listdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches


def wm_b_box_evaluation(model,watermark,key_embed,grad_select,sample_size,num_pixels,watermark_key_set_x,target_label,num_classes,step_size = 0.0001):
    input_shape = watermark_key_set_x.shape[1:]
    idx = random.sample(range(watermark_key_set_x.shape[0]),sample_size)
    grad_x = watermark_key_set_x[idx,::]

    y_true = K.placeholder(shape=model.output.shape)
    cross_ent = K.categorical_crossentropy(y_true,model.output)
    get_cross_ent = K.function([model.input,y_true],[cross_ent])

    estimated_grads = np.zeros((num_pixels,1))
    grad_select_indeces = []
    for index,selected in enumerate(K.get_value(grad_select).tolist()):
        if selected == 1.0:
            grad_select_indeces.append(int(index))

    for i,career_node in enumerate(grad_select_indeces):
        start_cross_ents = get_cross_ent([grad_x,keras.utils.to_categorical(target_label*np.ones(grad_x.shape[0],),num_classes)])[0]
        grad_x_moved = grad_x + K.get_value(K.reshape(step_size * K.cast(keras.utils.to_categorical([career_node],num_pixels),dtype=K.floatx()),shape=input_shape))
        end_cross_ents = get_cross_ent([grad_x_moved,keras.utils.to_categorical(target_label*np.ones(grad_x.shape[0],),num_classes)])[0]
        grads = (end_cross_ents-start_cross_ents)/step_size
        estimated_grads[career_node] = np.mean(grads)
    estimated_grads = K.variable(estimated_grads)
    projection = K.cast(K.reshape(0 <= (K.dot(key_embed,estimated_grads)),watermark.shape),K.floatx())
    matching_accuracy = K.get_value(1.00 - K.mean(K.abs(projection-watermark)))

    print('\nMatching_accuracy : ',matching_accuracy)
    return matching_accuracy

####################################################
####################################################
watermarked_model_file = '64-bit-CIFAR10.hdf5' #Change Correspondingly
#####################################
####### Preparing Dataset ###########
#####################################
sample_size = 50
target_class_label = 0.

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

idx_train = np.where(y_train[:]==target_class_label)[0]
watermarked_training_images = x_train[idx_train,::]
watermarked_training_images_label = y_train[idx_train,::]
print("watermark training size : ", watermarked_training_images.shape[0])

idx_test = np.where(y_test[:]==target_class_label)[0]
watermarked_testing_images = x_test[idx_test,::]
watermarked_testing_images_label = y_test[idx_test,::]
print("watermark testing size : ", watermarked_testing_images.shape[0])

# calculating pixel mean
x_train_mean = np.mean(x_train, axis=0)

watermarked_testing_images -=x_train_mean
watermarked_training_images -= x_train_mean
x_train -= x_train_mean
x_test -= x_train_mean

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
watermarked_testing_images_label = keras.utils.to_categorical(watermarked_testing_images_label,num_classes)
watermarked_training_images_label = keras.utils.to_categorical(watermarked_training_images_label,num_classes)

###########################################################################
num_pixels =  x_test.shape[1]*x_test.shape[2]*x_test.shape[3]
input_shape = x_test.shape[1:]
row,col,ch= input_shape
#################### Loading model and Watermark Info #####################
if not os.path.isfile(watermarked_model_file):
    print("Couldnt find the watermarked model!")

else:
    print("model exists...")
    watermark = K.variable(np.load('wm.npy'))
    print("watermark : ", K.get_value(watermark))
    key_embed = K.variable(np.load('embed_key_extended.npy'))
    grad_select = K.variable(np.load("grad_select.npy"))
    model = load_model(watermarked_model_file)
    opt = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
###########################################################################
if not os.path.isfile('INI_Results'+'.pkl'):
    noise_standard_variances = [0.0,0.001,0.005,0.01,0.02,0.05,0.1]
    trials = 20
    model_accuracy_for_noisy_images = []
    wm_accuracy_for_noisy_images = []
    for sigma in noise_standard_variances:
        wm_acc_list = []
        test_acc_list = []
        for t in range(trials):
            # had to load and flush due to memory bugs of keras.
            K.clear_session()
            watermark = K.variable(np.load('wm.npy'))
            key_embed = K.variable(np.load('embed_key_extended.npy'))
            grad_select = K.variable(np.load("grad_select.npy"))
            model = load_model(watermarked_model_file)
            opt = Adam(lr=0.0005)
            model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
            ###############################################################
            mean = 0.0
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy_watermarked_training_images = np.clip(watermarked_training_images + gauss + x_train_mean,0.0,1.0)-x_train_mean
            noisy_x_test = np.clip(x_test + gauss + x_train_mean,0.0,1.0)-x_train_mean
            test_acc = model.evaluate(noisy_x_test, y_test,verbose=0)[1]
            test_acc_list.append(test_acc)
            wm_acc = wm_b_box_evaluation(model=model,key_embed=key_embed,watermark=watermark,grad_select=grad_select,num_pixels=num_pixels,sample_size=sample_size,
                                        watermark_key_set_x=noisy_watermarked_training_images,target_label=target_class_label,num_classes=num_classes)
            wm_acc_list.append(wm_acc)
        
        wm_accuracy_for_noisy_images.append(sum(wm_acc_list)/len(wm_acc_list))
        model_accuracy_for_noisy_images.append(sum(test_acc_list)/len(test_acc_list))
        print("Test accuracy of the model in presence of input noise injection attack with std ",sigma, "is ","{0:.2f}".format(100*sum(test_acc_list)/len(test_acc_list)))
        print("Accuracy of extracted watermark in presence of input noise injection attack with std ",sigma, "is ","{0:.2f}".format(100*sum(wm_acc_list)/len(wm_acc_list)))

    with open('INI_Results'+'.pkl', 'wb') as f_pkl:
        pickle.dump(noise_standard_variances,f_pkl)
        pickle.dump(wm_accuracy_for_noisy_images,f_pkl)
        pickle.dump(model_accuracy_for_noisy_images,f_pkl)

else:
    with open('INI_Results'+'.pkl', 'rb') as f_pkl:
        noise_standard_variances = pickle.load(f_pkl)
        wm_accuracy_for_noisy_images = pickle.load(f_pkl)
        model_accuracy_for_noisy_images = pickle.load(f_pkl)



