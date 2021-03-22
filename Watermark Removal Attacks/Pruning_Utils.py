##### This code works on Keras version 2.2.4 with Tensorflow nightly version 1.14.1-dev20190402

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Layer , MaxPooling2D, Dropout, Input, concatenate, add, AveragePooling2D, Input, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.models import Model, load_model,Sequential
import numpy as np
from MaskedLayers import MaskedConv2D, MaskedDense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def PrunWeight(model, model_name, x_prune, y_prune, x_test, y_test, pruning_rate, compile_info , fine_tune):
    
    ############ Calculating weight limit for pruning ##### 
    ############ We do not consider biases in the pruning process #####
    parameters = []
    conv_layers_weights = []
    for layer in model.layers:
        if layer.get_config()['name'].find("conv") != -1:
            conv_layers_weights.append(layer.get_weights())

    for _, layer_weights in enumerate(conv_layers_weights):
        parameters.append(K.flatten(K.abs(layer_weights[0])))

    dense_layers_weights = []
    for layer in model.layers:
        if layer.get_config()['name'].find("dense") != -1:
            dense_layers_weights.append(layer.get_weights())

    for _, layer_weights in enumerate(dense_layers_weights):
        parameters.append(K.flatten(K.abs(layer_weights[0])))
    
    parameters =  K.concatenate(parameters)
    parameters = sorted(K.get_value(parameters).tolist())
    weight_limit = parameters[int(pruning_rate*len(parameters))]
    print("Pruning weight threshhold : ", weight_limit)
    ##################################################################
    dense_layers_weights = []
    conv_filter_weights = []
    batch_norm_params = []
    kernel_masks_for_dense_and_conv_layers = []
    model_tensors_dict = {}
    input_height,input_width,input_channels = model.input.shape[1:]

    pruned_model_input = Input(shape=(int(input_height),int(input_width),int(input_channels)))

    if model.layers[0].name.find('input') == -1:
        model_tensors_dict[str(model.layers[0].input.name)] = pruned_model_input
    else:
        model_tensors_dict[str(model.layers[0].output.name)] = pruned_model_input
        
    Flow = pruned_model_input
    
    for _,layer in enumerate(model.layers):
        if layer.get_config()['name'].find("conv2d") != -1:
            kernel_mask = K.cast(weight_limit <= K.abs(layer.get_weights()[0]) ,'float32')
            kernel_masks_for_dense_and_conv_layers.append(kernel_mask)
            Flow  = MaskedConv2D(filters=layer.get_config()['filters'], kernel_size=layer.get_config()['kernel_size'],kernel_initializer=layer.get_config()['kernel_initializer'], 
            kernel_regularizer= layer.get_config()['kernel_regularizer'], strides=layer.get_config()['strides'],
            padding=layer.get_config()['padding'], activation=layer.get_config()['activation'], use_bias=layer.get_config()['use_bias'], Masked=True , kernel_mask_val=kernel_mask)(model_tensors_dict[str(layer.input.name)])
            conv_filter_weights.append(layer.get_weights())
            model_tensors_dict[str(layer.output.name)] = Flow
            
        elif layer.get_config()['name'].find("dense") != -1:
            kernel_mask = K.cast(weight_limit <= K.abs(layer.get_weights()[0]) ,'float32')
            kernel_masks_for_dense_and_conv_layers.append(kernel_mask)
            Flow = MaskedDense(units=layer.get_config()['units'], activation=layer.get_config()['activation'],
            use_bias=layer.get_config()['use_bias'], kernel_initializer = layer.get_config()['kernel_initializer'],
            Masked=True , kernel_mask_val=kernel_mask)(model_tensors_dict[str(layer.input.name)])
            dense_layers_weights.append(layer.get_weights())
            model_tensors_dict[str(layer.output.name)] = Flow

        elif layer.get_config()['name'].find("activation") != -1:
            Flow = Activation.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow
            
        elif layer.get_config()['name'].find("max_pooling") != -1:
            Flow = MaxPooling2D.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow
        
        elif layer.get_config()['name'].find("average_pooling") != -1:
            Flow = AveragePooling2D.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow
        
        elif layer.get_config()['name'].find("dropout") != -1:
            Flow = Dropout.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow

        elif layer.get_config()['name'].find("flatten") != -1:
            Flow = Flatten.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow

        elif layer.get_config()['name'].find("add") != -1:
            input_tensors_list = []
            for idx in range(len(layer.input)):
                input_tensors_list.append(model_tensors_dict[layer.input[idx].name])
            Flow = add(input_tensors_list)
            model_tensors_dict[str(layer.output.name)] = Flow
        
        elif layer.get_config()['name'].find("batch_normalization") != -1:
            batch_norm_params.append(layer.get_weights())
            Flow = BatchNormalization.from_config(layer.get_config())(model_tensors_dict[str(layer.input.name)])
            model_tensors_dict[str(layer.output.name)] = Flow
        
        elif layer.get_config()['name'].find("input") != -1:
            pass
            
    pruned_model  = Model(pruned_model_input, Flow)
    ########################## setting the weight s of layers #############################
    for layer in pruned_model.layers:
        if layer.get_config()['name'].find("dense") != -1:
            pruned_weights = [dense_layers_weights[0][0]*K.get_value(kernel_masks_for_dense_and_conv_layers[0])]
            if layer.get_config()['use_bias']:
                pruned_weights.append(dense_layers_weights[0][1])
            layer.set_weights(pruned_weights)
            del kernel_masks_for_dense_and_conv_layers[0]
            del dense_layers_weights[0]
        
        elif layer.get_config()['name'].find("conv2d") != -1:
            pruned_weights = [conv_filter_weights[0][0]*K.get_value(kernel_masks_for_dense_and_conv_layers[0])]
            if layer.get_config()['use_bias']:
                pruned_weights.append(conv_filter_weights[0][1])
            layer.set_weights(pruned_weights)
            del kernel_masks_for_dense_and_conv_layers[0]
            del conv_filter_weights[0]
            
        elif layer.get_config()['name'].find("batch") != -1:
            layer.set_weights(batch_norm_params[0])
            del batch_norm_params[0]
    ############################### Fine-tuning ############################################
    pruned_model.compile(loss=compile_info['loss'],
            optimizer=compile_info['optimizer'],
            metrics=compile_info['metrics'])
    
    if not fine_tune:
        return pruned_model
    else:
        early_stopping = EarlyStopping(monitor='val_acc', patience=2,verbose=0)
        callbacks = [early_stopping]
        # fine-tuning the network.
        pruned_model.fit(x_prune, y_prune,
                    batch_size=256,
                    epochs=10,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=0
                    )

        return pruned_model




