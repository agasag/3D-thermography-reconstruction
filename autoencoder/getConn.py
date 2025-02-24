import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np

# siec z transferem wag pomiedzy enkoderem i dekoderem
def getConn(input_shape):

    input_img = layers.Input(shape=input_shape) 
    input_img_res = layers.ZeroPadding3D(padding = (14,14,7))(input_img)
    y = layers.Conv3D(32, (3, 3, 3), padding='same')(input_img_res)
    y = layers.LeakyReLU()(y)
    y = layers.MaxPool3D((2,2,2))(y)
    y = layers.Conv3D(64, (3, 3, 3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.MaxPool3D((2,2,2))(y)
    y1 = layers.Conv3D(128, (3, 3, 3), padding='same')(y) # skip-1
    y = layers.LeakyReLU()(y1)
    y = layers.MaxPool3D((2,2,2))(y)
    y = layers.Conv3D(256, (3, 3, 3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    y = layers.MaxPool3D((2,2,2))(y)
    y2 = layers.Conv3D(256, (3, 3, 3), padding='same')(y)# skip-2
    y = layers.LeakyReLU()(y2)
    y = layers.MaxPool3D((2,2,2))(y)
    y = layers.Conv3D(512, (3, 3, 3), padding='same')(y)
    y = layers.LeakyReLU()(y)
    #y = layers.MaxPool3D((2,2,2))(y)
    #y = layers.Conv3D(1024, (3, 3, 3), padding='same')(y)
    #y = layers.LeakyReLU()(y)#Flattening for the bottleneck
    ##y = layers.MaxPool3D((2,2,2))(y)
    vol = y.shape
    x = layers.Flatten()(y)
    latent = layers.Dense(128, activation='relu')(x)

    def lrelu_bn(inputs):
        lrelu = layers.LeakyReLU()(inputs)
        bn = layers.BatchNormalization()(lrelu)
        return bn
                
    y = layers.Dense(np.prod(vol[1:]), activation='relu')(latent)
    y = layers.Reshape((vol[1], vol[2], vol[3], vol[4]))(y)

    #y = layers.Conv3DTranspose(1024, (3,3,3), padding='same')(y)
    #y = layers.UpSampling3D((2,2,2))(y) 
    #y = layers.LeakyReLU()(y)

    y = layers.Conv3DTranspose(512,  (3,3,3), padding='same')(y)
    y = layers.UpSampling3D((2,2,2))(y) 
    y = layers.LeakyReLU()(y)

    y = layers.Conv3DTranspose(256,  (3,3,3), padding='same')(y)
    y = layers.Add()([y2, y]) # second skip connection added here
    y = layers.UpSampling3D((2,2,2))(y) 
    y = lrelu_bn(y)

    y = layers.Conv3DTranspose(256,  (3,3,3), padding='same')(y)
    y = layers.UpSampling3D((2,2,2))(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv3DTranspose(128,  (3,3,3), padding='same')(y)
    y = layers.Add()([y1, y]) # first skip connection added here
    y = layers.UpSampling3D((2,2,2))(y) 
    y = lrelu_bn(y)

    y = layers.Conv3DTranspose(64,  (3,3,3), padding='same')(y)
    y = layers.UpSampling3D((2,2,2))(y) 
    y = layers.LeakyReLU()(y)
    y = layers.Conv3DTranspose(1,  (3,3,3), activation='sigmoid', padding='same')(y)
    y = layers.Cropping3D(((14, 14, 7)))(y)

    model = Model(input_img, y)    
    model.build((100, 100, 50, 1))
    model.summary()
    #model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])

    return model