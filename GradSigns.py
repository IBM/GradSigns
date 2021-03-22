##### This code works on Keras version 2.2.4 with Tensorflow nightly version 1.14.1-dev20190402

from __future__ import print_function
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow 
from tensorflow import set_random_seed
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Layer , MaxPooling2D, Dropout
from keras.models import Sequential
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, load_model
from keras.datasets import cifar10
import numpy as np
import scipy.io as sio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random,argparse,pickle, csv,time
###################################
random.seed(int(time.time()))
set_random_seed(int(time.time()))
##################################
###### Model's Architecture ######
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,kernel_initializer='he_normal')(y)
    outputs = Activation('softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
######  Dataset related methods ###### 
def equal_split(X,Y,train_set_sample_per_label,number_of_labels): ## Creating test and training sets with certain number of samples from each classification label in them. 
    x_train_split = []
    y_train_split = []

    x_test_split = []
    y_test_split = []
    
    for label in range(number_of_labels):
        
        max_sample_size = len(list(np.where(Y==label)[0]))
        train_idx = random.sample(list(np.where(Y==label)[0]),min(train_set_sample_per_label,max_sample_size))
        test_idx = list(set(np.where(Y==label)[0]).difference(set(train_idx)))

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

def image_generator(train_x, train_label, watermark_train_x,watermark_train_y, target_class_imgs_per_batch ,total_batch_size):
    
	while True:
		
		idx = random.sample(range(0,watermark_train_x.shape[0]),target_class_imgs_per_batch)
		batch_x = watermark_train_x[idx,::]
		batch_y = watermark_train_y[idx,::]

		idx = random.sample(range(0,train_x.shape[0]),total_batch_size-target_class_imgs_per_batch)
		batch_x = np.vstack((batch_x,train_x[idx,::]))
		batch_y = np.vstack((batch_y,train_label[idx,::]))
		
		yield ( batch_x, batch_y )

def train_image_generator(generator, train_x, train_label, watermark_train_x,watermark_train_y, target_class_imgs_per_batch ,total_batch_size):
    
    wm_batches = generator.flow(watermark_train_x,watermark_train_y, batch_size=target_class_imgs_per_batch)
    train_batches = generator.flow(train_x, train_label, batch_size=total_batch_size-target_class_imgs_per_batch) 
	
    while True:
        wm_batch_x,wm_batch_y = wm_batches.next()
        train_batch_x,train_batch_y = train_batches.next()

        batch_x = np.vstack((wm_batch_x,train_batch_x))
        batch_y = np.vstack((wm_batch_y,train_batch_y))
        
        yield ( batch_x, batch_y )

###### Training phase related methods ###### 
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def key_extend(key,dim,selected_idx): ## Converting dimensionality of generated embedding key to mathch dimensionality of model's input by padding zeros to be used during model training.  
	Extended_Key = np.zeros((key.shape[0],dim))
	indeces_sorted = sorted(selected_idx)
	for row in range(key.shape[0]):
		j=0
		for bit in indeces_sorted:
			Extended_Key[row][bit] = key[row][j]
			j+=1
	return K.variable(Extended_Key)

def reg_term_wrapper(model,regularizer_coefficient,watermark,key_embed,grad_select,target_class,num_classes,num_pixels):
	def reg_term_value(y_true, y_pred):
		target_class_one_hot = K.variable(keras.utils.to_categorical([target_class],num_classes))
		label_mask = K.cast(K.all(K.equal(y_true,target_class_one_hot),axis=-1),K.floatx())
		cross_ent = K.categorical_crossentropy(y_true, y_pred)
		grads = K.gradients(K.sum(cross_ent), model.input)[0]
		label_masked_grads = K.switch(1.0-label_mask,K.zeros_like(grads),grads)	
		mean_grads = K.sum(label_masked_grads,axis=0)/K.sum(label_mask)
		flattened_mean_grads = K.flatten(mean_grads)
		selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
		projection = K.reshape(K.sigmoid(K.dot(key_embed,selected_mean_grads)),watermark.shape)
		reg = keras.losses.binary_crossentropy(watermark,projection)
		return reg
	return reg_term_value

def wm_acc_wrapper(model,regularizer_coefficient,watermark,key_embed,grad_select,target_class,num_classes,num_pixels):
	def wm_acc(y_true, y_pred):
		target_class_one_hot = K.variable(keras.utils.to_categorical([target_class],num_classes))
		label_mask = K.cast(K.all(K.equal(y_true,target_class_one_hot),axis=-1),K.floatx())
		cross_ent = K.categorical_crossentropy(y_true, y_pred)
		grads = K.gradients(K.sum(cross_ent), model.input)[0]
		label_masked_grads = K.switch(1.0-label_mask,K.zeros_like(grads),grads)	
		mean_grads = K.sum(label_masked_grads,axis=0)/K.sum(label_mask)
		flattened_mean_grads = K.flatten(mean_grads)
		selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
		projection = K.cast(K.reshape(0 <= (K.dot(key_embed,selected_mean_grads)),watermark.shape),K.floatx())
		matching_accuracy = 1.00 - K.mean(K.abs(projection-watermark))
		return matching_accuracy
	return wm_acc

def progress_wrapper(model,regularizer_coefficient,watermark,key_embed,grad_select,target_class,num_classes,num_pixels):
	def progress(y_true, y_pred):
		target_class_one_hot = K.variable(keras.utils.to_categorical([target_class],num_classes))
		label_mask = K.cast(K.all(K.equal(y_true,target_class_one_hot),axis=-1),K.floatx())
		cross_ent = K.categorical_crossentropy(y_true, y_pred)
		grads = K.gradients(K.sum(cross_ent), model.input)[0]
		label_masked_grads = K.switch(1.0-label_mask,K.zeros_like(grads),grads)	
		mean_grads = K.sum(label_masked_grads,axis=0)/K.sum(label_mask)
		flattened_mean_grads = K.flatten(mean_grads)
		selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
		projection = K.cast(K.reshape(0 <= (K.dot(key_embed,selected_mean_grads)),watermark.shape),K.floatx())
		matching_accuracy = 1.00 - K.mean(K.abs(projection-watermark))
		accuracy = keras.metrics.categorical_accuracy(y_true,y_pred)
		return 0.5*(matching_accuracy)+accuracy
	return progress

def GradSigns_loss_wrapper(model,regularizer_coefficient,watermark,key_embed,grad_select,target_class,num_classes,num_pixels):
	def GradSigns_loss(y_true, y_pred):
		target_class_one_hot = K.variable(keras.utils.to_categorical([target_class],num_classes))
		label_mask = K.cast(K.all(K.equal(y_true,target_class_one_hot),axis=-1),K.floatx())
		cross_ent = K.categorical_crossentropy(y_true, y_pred)
		grads = K.gradients(K.sum(cross_ent), model.input)[0]
		label_masked_grads = K.switch(1.0-label_mask,K.zeros_like(grads),grads)	
		mean_grads = K.sum(label_masked_grads,axis=0)/K.sum(label_mask)
		flattened_mean_grads = K.flatten(mean_grads)
		selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1)) 
		projection = K.reshape(K.sigmoid(K.dot(key_embed,selected_mean_grads)),watermark.shape)
		reg = keras.losses.binary_crossentropy(watermark,projection)
		return cross_ent + regularizer_coefficient*reg 
	return GradSigns_loss

#####################################################
############ Training parameters ####################
#####################################################
train_batch_size = 100 
target_images_per_training_batch = 10
train_epochs = 200
#####################################################
############ Watermarking parameters ################
#####################################################
target_class_label = 0.
sample_size = 50
####################################################
if __name__== "__main__":
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--trial', action="store", dest='t', default=1)
    parser.add_argument('--carriers', action="store", dest='c', default=64)
    parser.add_argument('--wm', action="store", dest='wm', default=16)
    parser.add_argument('--lambda', action="store", dest='l', default=0.3)
    args = parser.parse_args()
    trial_number = int(args.t)
    career_nodes_number = int(args.c) 
    regularizer_coefficient = float(args.l)
    watermark_length = int(args.wm)
    print("Watermark length : ", watermark_length)
    print('Trial number ', trial_number, 'with ', career_nodes_number ,' career nodes.')
    print('regularizer coefficient : ', regularizer_coefficient)
    #############################################################################
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    idx_train = np.where(y_train[:]==target_class_label)[0]
    watermark_train_x = x_train[idx_train,::]
    watermark_train_y = y_train[idx_train,::]
    print("watermark training size : ", watermark_train_x.shape[0])

    idx_test = np.where(y_test[:]==target_class_label)[0]
    watermark_test_x = x_test[idx_test,::]
    watermark_test_y = y_test[idx_test,::]
    print("watermark testing size : ", watermark_test_x.shape[0])

    # removing target class instances from testing set as we inted to use the genrator to ensure certain number of target images exist in each batch
    x_test_wo_wm = np.delete(x_test, idx_test, axis=0)
    y_test_wo_wm = np.delete(y_test, idx_test, axis=0)

    # calculating pixel mean
    x_train_mean = np.mean(x_train, axis=0)

    watermark_test_x -=x_train_mean
    watermark_train_x -= x_train_mean
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_test_wo_wm -= x_train_mean

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_test_wo_wm = keras.utils.to_categorical(y_test_wo_wm,num_classes)
    watermark_test_y = keras.utils.to_categorical(watermark_test_y,num_classes)
    watermark_train_y = keras.utils.to_categorical(watermark_train_y,num_classes)

    DataGen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)
        
    DataGen.fit(x_train)
    # removing target class instances from traning set as we inted to use the genrator to ensure certain number of target images in each batch
    x_train = np.delete(x_train, idx_train, axis=0)
    y_train = np.delete(y_train, idx_train, axis=0)
    ######################## creating model ####################################
    input_shape = x_train.shape[1:]
    num_pixels =  x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
    model_type = str(watermark_length)+'_bit_CIFAR10'
    depth = 20
    model = resnet_v1(input_shape=input_shape, depth=depth)
    print(model.summary())
    ################################Loading WM Info ####################################
    if not os.path.isdir('./'+str(watermark_length)):
        os.system('mkdir ' + './'+str(watermark_length))
    if not os.path.isdir('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)):
        os.system('mkdir '+'./'+str(watermark_length)+'/CN-'+str(career_nodes_number))
    if not os.path.isdir('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)):
        os.system('mkdir '+'./'+str(watermark_length)+'/CN-'+str(career_nodes_number))

    ####################################################################################
    if not os.path.isfile('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'wm.npy'):
        watermark = K.cast( 0.5 <= K.random_uniform_variable(shape=(watermark_length,), low=0, high=1),'float32')
        grad_select = [0 for _ in range(num_pixels)]
        selected_idx = random.sample(range(num_pixels),career_nodes_number)
        for i in selected_idx:
            grad_select[i]=1
        grad_select = np.array(grad_select)
        grad_select = K.variable(grad_select)

        key_embed = K.random_uniform_variable(shape=(watermark_length,career_nodes_number), low=-1, high=1)
        key_embed_extended = key_extend(K.get_value(key_embed),num_pixels,selected_idx)
        np.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'wm.npy',K.get_value(watermark))
        np.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+"embed_key.npy",K.get_value(key_embed))
        np.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+"embed_key_extended.npy",K.get_value(key_embed_extended))
        np.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'grad_select.npy',K.get_value(grad_select))
        key_embed = key_embed_extended
        print("key_embed dimensions : ", key_embed.shape)
    
    else:
        print('found watermark info!')
        watermark = K.variable(np.load('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'wm.npy'))
        print("watermark : ", K.get_value(watermark))
        key_embed = K.variable(np.load('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'embed_key_extended.npy'))
        grad_select = K.variable(np.load('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+"grad_select.npy"))
    ####################################################################################
    # Run training.
    if not os.path.isfile('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5'):
        opt = Adam(lr=lr_schedule(0))
        model.compile(loss=GradSigns_loss_wrapper(model=model, regularizer_coefficient=regularizer_coefficient, watermark = watermark, key_embed = key_embed, grad_select = grad_select, target_class = target_class_label, num_classes =num_classes, num_pixels = num_pixels),
                    optimizer=opt,
                    metrics=['accuracy',reg_term_wrapper(model=model, regularizer_coefficient=regularizer_coefficient, watermark = watermark, key_embed = key_embed, grad_select = grad_select, target_class = target_class_label, num_classes =num_classes, num_pixels = num_pixels),
                wm_acc_wrapper(model=model, regularizer_coefficient=regularizer_coefficient, watermark = watermark, key_embed = key_embed, grad_select = grad_select, target_class = target_class_label, num_classes =num_classes, num_pixels = num_pixels),
                progress_wrapper(model=model, regularizer_coefficient=regularizer_coefficient, watermark = watermark, key_embed = key_embed, grad_select = grad_select, target_class = target_class_label, num_classes =num_classes, num_pixels = num_pixels)])

        print("Starting the model training phase.")

        pr_checkpoint = ModelCheckpoint(filepath='./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5',monitor='val_progress',verbose=1,save_best_only=True, mode='max')
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
        callbacks = [pr_checkpoint,lr_scheduler, lr_reducer]

        data_gen = train_image_generator(DataGen,x_train,y_train, watermark_train_x,watermark_train_y, target_class_imgs_per_batch=target_images_per_training_batch,total_batch_size=train_batch_size)
        data_gen_val = image_generator(x_test_wo_wm,y_test_wo_wm, watermark_test_x,watermark_test_y, target_class_imgs_per_batch=50,total_batch_size=450)
        
        history = model.fit_generator(
            generator=data_gen,
            steps_per_epoch=max(int((x_train.shape[0]+watermark_train_x.shape[0])/train_batch_size),50),
            epochs=train_epochs,
            validation_data=data_gen_val,
            validation_steps=int((x_test_wo_wm.shape[0]+watermark_test_x.shape[0])/500),
            shuffle=True,
            callbacks=callbacks)

        with open('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'history_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.pkl','wb') as f_pkl:
            pickle.dump(history,f_pkl)
        
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        model.load_weights('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5')
        model.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5')

    else:
        opt = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        model.load_weights('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5')
        model.save('./'+str(watermark_length)+'/CN-'+str(career_nodes_number)+'/'+'Watermarked_'+model_type+'_'+str(regularizer_coefficient)+'_'+str(trial_number)+'.hdf5')
        
    test_acc_after = model.evaluate(x_test, y_test,verbose=0)[1]          
    print("Model accuracy after embedding the watermark: ", test_acc_after)
    ########################### Testing the Accuracy of Watermark ############################
    y_true = K.placeholder(shape=model.output.shape)
    cross_ent = K.categorical_crossentropy(y_true,model.output)
    get_grads = K.function([model.input,y_true],K.gradients(cross_ent,model.input))

    matching_accuracy_list = []
    for _ in range(20):
        idx = random.sample(range(watermark_train_x.shape[0]),sample_size)

        grad_x = watermark_train_x[idx,::]
        grad_y = watermark_train_y[idx,::]

        gradients = K.variable(get_grads([grad_x,grad_y])[0])
        mean_gradients = K.mean(gradients,axis=0)
        flattened_mean_grads = K.flatten(mean_gradients)
        selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
        projection = K.cast(K.reshape(0 <= (K.dot(key_embed,selected_mean_grads)),watermark.shape),K.floatx())

        matching_accuracy = 1.00 - K.mean(K.abs(projection-watermark))
        matching_accuracy_list.append(K.get_value(matching_accuracy))
    matching_acc_train = sum(matching_accuracy_list)/len(matching_accuracy_list)
    print('matching_accuracy : ', matching_acc_train)

    matching_accuracy_list = []
    for _ in range(20):
        idx = random.sample(range(watermark_test_x.shape[0]),sample_size)

        grad_x = watermark_test_x[idx,::]
        grad_y = watermark_test_y[idx,::]

        gradients = K.variable(get_grads([grad_x,grad_y])[0])
        mean_gradients = K.mean(gradients,axis=0)
        flattened_mean_grads = K.flatten(mean_gradients)
        selected_mean_grads = K.reshape(K.switch(1.0-grad_select,K.zeros_like(flattened_mean_grads),flattened_mean_grads),shape=(num_pixels,1))
        projection = K.cast(K.reshape(0 <= (K.dot(key_embed,selected_mean_grads)),watermark.shape),K.floatx())

        matching_accuracy = 1.00 - K.mean(K.abs(projection-watermark))
        matching_accuracy_list.append(K.get_value(matching_accuracy)) 
    matching_acc_test = sum(matching_accuracy_list)/len(matching_accuracy_list)
    print('matching_accuracy : ', matching_acc_test)