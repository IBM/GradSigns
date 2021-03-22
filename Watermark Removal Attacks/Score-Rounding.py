
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
import os,psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import scipy.io as sio
import random,pickle
from os import listdir
from os.path import isfile, join

pid = os.getpid()
py = psutil.Process(pid)

def wm_b_box_evaluation_with_output_rounding(step_size,reported_precision,model,watermark,key_embed,grad_select,sample_size,num_pixels,watermark_key_set_x,target_label,num_classes):
    input_shape = watermark_key_set_x.shape[1:]
    idx = random.sample(range(watermark_key_set_x.shape[0]),sample_size)
    grad_x = watermark_key_set_x[idx,::]

    y_true = K.placeholder(shape=model.output.shape)
    model_output = K.placeholder(shape=model.output.shape)
    cross_ent = K.categorical_crossentropy(y_true,model_output)
    get_cross_ent = K.function([model_output,y_true],[cross_ent])
    
    estimated_grads = np.zeros((num_pixels,1))
    grad_select_indeces = []
    number_of_career_nodes = int(sum(K.get_value(grad_select).tolist()))
    for index,selected in enumerate(K.get_value(grad_select).tolist()):
        if selected == 1.0:
            grad_select_indeces.append(int(index))

    start_predicted_probs = model.predict(grad_x)
    for i,career_node in enumerate(grad_select_indeces):
        
        rounded_predicted_probs = np.around(start_predicted_probs,reported_precision)
        start_cross_ents = get_cross_ent([rounded_predicted_probs,keras.utils.to_categorical(target_label*np.ones(grad_x.shape[0],),num_classes)])[0]

        grad_x_moved = grad_x + K.get_value(K.reshape(step_size * K.cast(keras.utils.to_categorical([career_node],num_pixels),dtype=K.floatx()),shape=input_shape))
        end_predicted_probs = model.predict(grad_x_moved)
        rounded_predicted_probs = np.around(end_predicted_probs,reported_precision)
        end_cross_ents = get_cross_ent([rounded_predicted_probs,keras.utils.to_categorical(target_label*np.ones(grad_x.shape[0],),num_classes)])[0]
        grads = (end_cross_ents-start_cross_ents)/step_size
        estimated_grads[career_node] = np.mean(grads)
        ###########################
        end_predicted_probs = None
        end_cross_ents = None
        rounded_predicted_probs = None
        start_cross_ents = None
        print("Memory usage : ", py.memory_info()[0]//(10**6),' career node [',i,'/', number_of_career_nodes,']',end='\r')

    
    estimated_grads = K.variable(estimated_grads)
    projection = K.cast(K.reshape(0 <= (K.dot(key_embed,estimated_grads)),watermark.shape),K.floatx())
    matching_accuracy = K.get_value(1.00 - K.mean(K.abs(projection-watermark)))
    
    # Memory Release
    estimated_grads = None
    start_predicted_probs = None
    end_predicted_probs = None
    grad_select_indeces = None
    grad_x = None
    projection = None

    print('\nMatching_accuracy : ',matching_accuracy)
    return matching_accuracy


#####################################
#####################################
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
############################################################
num_pixels =  x_test.shape[1]*x_test.shape[2]*x_test.shape[3]
input_shape = x_test.shape[1:]
row,col,ch= input_shape
#################### Loading model and Watermark Info #####################
if not os.path.isfile(watermarked_model_file):
    print("Couldnt find the watermarked model!")

else:
    print("Model exists...")
    watermark = K.variable(np.load('wm.npy'))
    print("Watermark : ", K.get_value(watermark))
    key_embed = K.variable(np.load('embed_key_extended.npy'))
    grad_select = K.variable(np.load("grad_select.npy"))
    model = load_model(watermarked_model_file)
    opt = Adam(lr=0.0005)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
###########################################################################
if not os.path.isfile('OR-Results.pkl'):
    precisions = [4,3,2,1]
    step_sizes = [1e-5,1e-4,1e-3,1e-2,1e-1]
    trials = 5
    wm_acc_stats = {}
    for p in precisions:
        print("########### precision: ",p, ' ############' )
        for s in step_sizes:
            print("------------- step size: ",s, ' ------------' )
            wm_list = []
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
                wm_acc = wm_b_box_evaluation_with_output_rounding(step_size=s,reported_precision=p,model=model,key_embed=key_embed,watermark=watermark,grad_select=grad_select,num_pixels=num_pixels,sample_size=sample_size,watermark_key_set_x=watermarked_training_images,target_label=target_class_label,num_classes=num_classes)
                wm_list.append(wm_acc)
            
            wm_acc = sum(wm_list)/len(wm_list)
            if not str(p) in wm_acc_stats:
                wm_acc_stats[str(p)]=[(s,wm_acc)]
            else:
                wm_acc_stats[str(p)].append((s,wm_acc))

            with open('OR-Results.pkl', 'wb') as f_pkl:
                pickle.dump(wm_acc_stats,f_pkl)
                
    
else:
    with open('OR-Results.pkl', 'rb') as f_pkl:
        wm_acc_stats = pickle.load(f_pkl)




