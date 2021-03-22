##### This code works on Keras version 2.2.4 with Tensorflow nightly version 1.14.1-dev20190402

import keras
from keras.optimizers import Adam,SGD
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar10,mnist
import numpy as np
import random
import csv
import argparse
import sys
from Pruning_Utils import PrunWeight

def equal_split(X,Y,train_set_sample_per_label,number_of_labels):
    x_train_split = []
    y_train_split = []

    x_test_split = []
    y_test_split = []
    
    for label in range(number_of_labels):
        max_sample_size = len(list(np.where(np.argmax(Y,axis=1)[:]==label)[0]))
        train_idx = random.sample(list(np.where(np.argmax(Y,axis=1)[:]==label)[0]),min(train_set_sample_per_label,max_sample_size))
        test_idx = list(set(np.where(np.argmax(Y,axis=1)[:]==label)[0]).difference(set(train_idx)))

        train_images = X[train_idx,::]
        train_labels = Y[train_idx,::]
        
        test_images = X[test_idx,::]
        test_labels = Y[test_idx,::]
        
        if label == 0 :
            x_train_split = train_images
            y_train_split = train_labels
            x_test_split = test_images
            y_test_split = test_labels
        else:
            x_train_split = np.vstack((x_train_split,train_images))
            y_train_split = np.vstack((y_train_split,train_labels))
            x_test_split = np.vstack((x_test_split,test_images))
            y_test_split = np.vstack((y_test_split,test_labels))
    
    return x_train_split, x_test_split, y_train_split, y_test_split
    
def performance_check(model, grad_select, watermark, key_embed, watermarked_training_images, watermarked_training_images_label,num_pixels):
    ###############################################################################
    y_true = K.placeholder(shape=model.output.shape)
    cross_ent = K.categorical_crossentropy(y_true,model.output)
    get_grads = K.function([model.input,y_true],K.gradients(cross_ent,model.input))
    ################################################################################
    trial_results = []
    for _ in range(5):
        idx = random.sample(range(watermarked_training_images.shape[0]),50)
        grad_x = watermarked_training_images[idx,::]
        grad_y = watermarked_training_images_label[idx,::]
        gradients = K.variable(get_grads([grad_x,grad_y])[0])
        mean_gradients = K.mean(gradients,axis=0)
        flattened_mean_grads = K.flatten(mean_gradients)
        selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
        projection = K.cast(K.reshape(0 <= (K.dot(key_embed,selected_mean_grads)),watermark.shape),K.floatx())
        wm_tr_acc = 1.00 - K.mean(K.abs(projection-watermark))
        trial_results.append(K.get_value(wm_tr_acc))
    ############################################################################################################
    mean_wm_acc = sum(trial_results)/len(trial_results)
    return mean_wm_acc


if __name__== "__main__":

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--lr', action="store", dest='lr', default=-1)
    parser.add_argument('--avd', action="store", dest='avd', default=-1)
    args = parser.parse_args()

    learning_rate = float(args.lr)
    num_available_samples_per_label = int(args.avd)
    print("learning_rate : ", learning_rate)
    print("samples avaialable per label: ",num_available_samples_per_label )

    target_class_label = 0.

    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_pixels =  x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
    input_shape = x_train.shape[1:]
    ##################### Creating the watermark training images #########
    idx = np.where(y_train[:]==target_class_label)[0]
    watermarked_training_images = x_train[idx,::]
    watermarked_training_images_label = y_train[idx,::]
    ###################### Creating the watermark testing images ###########
    idx = np.where(y_test[:]==target_class_label)[0]
    watermarked_testing_images = x_test[idx,::]
    watermarked_testing_images_label = y_test[idx,::]
    #######################################################################
    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    watermarked_testing_images = watermarked_testing_images.astype('float32') / 255
    watermarked_training_images = watermarked_training_images.astype('float32') / 255
    
    # subtracting pixel mean
    x_train_mean = np.mean(x_train, axis=0)

    x_train -= x_train_mean
    x_test -= x_train_mean
    watermarked_training_images -= x_train_mean
    watermarked_testing_images -= x_train_mean

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    watermarked_training_images_label = keras.utils.to_categorical(watermarked_training_images_label,num_classes)
    watermarked_testing_images_label = keras.utils.to_categorical(watermarked_testing_images_label,num_classes)

    # Dataset avaialbe for the Adversary
    available_x, remaining_x, available_y, remaining_y = equal_split(X=x_train, Y=y_train, train_set_sample_per_label=num_available_samples_per_label,number_of_labels=num_classes)
    x_train = None
    y_train = None
    user_x_train, user_x_test, user_y_train, user_y_test = equal_split(X=available_x, Y=available_y, train_set_sample_per_label=int(num_available_samples_per_label*0.70),number_of_labels=num_classes)
    available_x=None
    available_y=None
    ##############################################################################################################3
    watermarked_model_file = '64-bit-CIFAR10' 
    
    model = load_model(watermarked_model_file+".hdf5",compile=False)
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=learning_rate, decay=1e-6),
            metrics=['accuracy'])
    
    watermark = K.variable(np.load('wm.npy'))
    key_embed = K.variable(np.load('embed_key_extended.npy'))
    grad_select = K.variable(np.load("grad_select.npy"))
    
    wm_acc = performance_check(model = model, grad_select = grad_select, watermark = watermark,key_embed = key_embed,
                                                    watermarked_training_images = watermarked_training_images,watermarked_training_images_label = watermarked_training_images_label
                                                    ,num_pixels = num_pixels)
    base_line = model.evaluate(x_test,y_test,verbose=0)[1]


    print("Model test accuracy: ", base_line)
    print("Model WM accuracy: ", wm_acc)

    compile_info = {}
    compile_info['loss'] = 'categorical_crossentropy'
    compile_info['optimizer'] = Adam(lr=learning_rate, decay=1e-6)
    compile_info['metrics'] = ['accuracy']

    for pr_rate in range(0,10):
        print("------------------------ pruning rate : ", "{0:0.2f}".format(pr_rate/10), "------------------------------")
        pruned_model =  PrunWeight(model=model, model_name=watermarked_model_file, x_prune=user_x_train, y_prune=user_y_train, x_test=user_x_test ,y_test=user_y_test, pruning_rate=pr_rate/10, compile_info=compile_info,
                    fine_tune=True)
        test_acc = pruned_model.evaluate(x_test,y_test,verbose=0)[1]
        wm_acc = performance_check(model = pruned_model, grad_select = grad_select, watermark = watermark,key_embed = key_embed,
                                            watermarked_training_images = watermarked_training_images,watermarked_training_images_label = watermarked_training_images_label,num_pixels = num_pixels)
  
        print("Pruned watermarked model test accuracy: ", test_acc)
        print("Pruned watermarked model wm accuracy: ",wm_acc)
