#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2
import pylab as pla
import re
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Deconv2D, UpSampling2D, Deconvolution2D, Flatten, ZeroPadding2D, merge
from keras.models import Sequential
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import callbacks


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
ktf.set_session(session)

batch_size = 16
h = 240
w = 320
channel = 3
m_ori = 6
n = 30 * 40 * 8
dim = 80 * 60 * 3
hidden_h = 30
hidden_w = 40
hidden_c = 8
hidden_dim1 = 256
hidden_dim = 128
epsilon_std = 1.0
use_loss = 'mse' # 'mse' or 'xent'
decay = 1e-4 # weight decay, a.k. l2 regularization
use_bias = True


np.random.seed(1111)  

############################################# original network #######################################
## Encoder
#import pdb;pdb.set_trace()
leakyrelu = LeakyReLU(alpha=0.05)
x = Input(shape=(h, w, channel))
cnn_1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN1')(x)
cnn_1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN2')(cnn_1)
max_1 = MaxPooling2D(pool_size=(2, 2), border_mode='same', name='max1')(cnn_1)#(?, 120, 160, 64) 
#pad_2 = ZeroPadding2D(padding=(0, 1))(max_1) #This is used in local detection

cnn_2 = Conv2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN3')(max_1)#pad_2  #(?, 120, 160, 32) 
cnn_2 = Conv2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN4')(cnn_2)
max_2 = MaxPooling2D(pool_size=(2, 2), border_mode='same', name='max2')(cnn_2)#(?, 60, 80, 32) 


cnn_3 = Conv2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN5')(max_2)#(?, 60, 80, 16)
cnn_3 = Conv2D(nb_filter=16, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN6')(cnn_3)
max_3 = MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='max3')(cnn_3)#(?, 30, 40, 16)


cnn_4 = Conv2D(nb_filter=8, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN7')(max_3)
cnn_4 = Conv2D(nb_filter=8, nb_row=3, nb_col=3, border_mode='same', activation='relu', name='CNN8')(cnn_4)#(?, 30, 40, 8) 

flat = Flatten()(cnn_4)
h_encoded = Dense(hidden_dim1, W_regularizer=l2(decay), b_regularizer=l2(decay),  activation='relu', name='Dense1_vae')(flat)  #use_bias=use_bias,
h_encoded = Dense(hidden_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), activation='relu', name='Dense2_vae')(h_encoded)
z_mean = Dense(m_ori, W_regularizer=l2(decay), b_regularizer=l2(decay), name='Dense_mean_vae')(h_encoded)
z_log_var = Dense(m_ori, W_regularizer=l2(decay), b_regularizer=l2(decay), name='Dense_sigma_vae')(h_encoded)

## Sampler
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal_variable(shape=(m_ori, ), mean=0.,
                                       scale=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(m_ori,), name='Lambda')([z_mean, z_log_var])#(?, 6)

# Initialize
decoder_h1 = Dense(hidden_dim, W_regularizer=l2(decay), b_regularizer=l2(decay), activation='relu', name='Dense3_vae')
decoder_h = Dense(hidden_dim1, W_regularizer=l2(decay), b_regularizer=l2(decay), activation='relu', name='Dense4_vae')
decoder_mean = Dense(n, W_regularizer=l2(decay), b_regularizer=l2(decay),  activation='tanh', name='Dense5_vae')

## Decoder
h_decoded1 = decoder_h1(z)
h_decoded = decoder_h(h_decoded1)
x_hat = decoder_mean(h_decoded)


def dense2conv(args):
    x_hat = args
    return K.reshape(x_hat, (-1, hidden_h, hidden_w, hidden_c))

dense_out = Lambda(dense2conv, name='Lambda2')(x_hat)#(?, 30, 40, 8)


mer_5 = merge([Conv2D(16, 2, 2, activation='relu', border_mode='same', name='mer5_CNN')(UpSampling2D(size=(2, 2), name='upsam_5')(dense_out)), cnn_3], 
		 mode='concat', concat_axis=3, name='mer_5')#(?, 60, 80, 32)
cnn_5 = Conv2D(32, 3, 3, activation='relu', border_mode='same', name='cnn5')(mer_5)
cnn_5 = Conv2D(32, 3, 3, activation='relu', border_mode='same', name='cnn5.2')(cnn_5)

mer_6 = merge([Conv2D(32, 2, 2, activation='relu', border_mode='same', name='mer6_CNN')(UpSampling2D(size=(2, 2), name='upsam_6')(cnn_5)), cnn_2],
         mode='concat', concat_axis=3, name='mer_6')#(?, 120, 160, 64) 
cnn_6 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='cnn6')(mer_6)
cnn_6 = Conv2D(64, 3, 3, activation='relu', border_mode='same', name='cnn6.2')(cnn_6)

mer_7 = merge([Conv2D(64, 2, 2, activation='relu', border_mode='same', name='mer7_CNN')(UpSampling2D(size=(2, 2), name='upsam_7')(cnn_6)), cnn_1],
            mode='concat', concat_axis=3, name='mer_7')#(?, 240, 320, 128)
cnn_7 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='cnn7')(mer_7)
cnn_7 = Conv2D(128, 3, 3, activation='relu', border_mode='same', name='cnn7.2')(cnn_7)

conv10 = Conv2D(channel, 1, 1, activation='relu', name='cnn8')(cnn_7)#(?, 240, 320, 1) 

## loss
def vae_loss(x, conv10):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    xent_loss = objectives.binary_crossentropy(x, conv10)
    mse_loss = K.mean(K.square(x - conv10), axis=[1, 2, 3])
    mse_kl_loss = mse_loss + kl_loss
    if use_loss == 'xent':
        return xent_loss + kl_loss
    elif use_loss == 'mse':
        return mse_kl_loss


vae = Model(x, conv10)
vae.summary()

adam = Adam(lr=0.0001)
vae.compile(optimizer=adam, loss=vae_loss)
##------------------------------------------------------------------------------------------------------------

checkpoint = ModelCheckpoint("./weight/loss_{loss_ty}".format(loss_ty=use_loss)+"_weight_{epoch:02d}-{val_loss:.15f}-{loss:.15f}.hdf5", verbose=0, save_best_only=True)

early_check = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')

board_check = TensorBoard(log_dir='./tmp/log')

vae.fit(ntrain, ntrain,
        shuffle=True,
        nb_epoch=1000,
        batch_size=batch_size,
        validation_data=(ntest1, ntest1),
        verbose=2,
        callbacks=[checkpoint, early_check, board_check])

##-----------------------------------------------------------------------------------------------------------------













