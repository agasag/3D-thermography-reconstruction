import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class TemperatureAutoencoder(Model):
    def __init__(self, input_shape):
        super(TemperatureAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.ZeroPadding2D(padding=14),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool2D((2, 2)),

            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool2D((2, 2)),

            # layers.Flatten(),
            layers.GlobalMaxPooling2D(),
            layers.Dense(8 * 8 * 256),
            layers.Reshape((8, 8, 256))
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(input_shape[2], kernel_size=3, padding='same'),
            layers.Cropping2D(cropping=14)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TemperatureAutoencoder3D(Model):
    def __init__(self, input_shape):
        super(TemperatureAutoencoder3D, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv3D(64, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Conv3D(32, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 1)),

            layers.Conv3D(1, (3, 3, 3), padding='same'),

            layers.Flatten(),
            layers.Dense(500),
            layers.Dense(25 * 25 * 25),
            layers.Reshape((25, 25, 25, 1))
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 1), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2),  padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv3DTranspose(1, (3, 3, 3), padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TemperatureAutoencoderMix(Model):
    def __init__(self, input_shape):
        super(TemperatureAutoencoderMix, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv3D(64, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Conv3D(1, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Reshape((50, 50, 25)),

            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool2D((2, 2))
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(25, (2, 2), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((50, 50, 25, 1)),
            layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2),  padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # layers.Conv3DTranspose(1, (3, 3, 3), padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class TemperatureAutoencoderAS(Model):
    def __init__(self, input_shape):
        super(TemperatureAutoencoderAS, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.ZeroPadding3D(padding = (14,14,7)),
            layers.Conv3D(32, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Conv3D(64, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Conv3D(128, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Conv3D(256, (3, 3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.MaxPool3D((2, 2, 2)),

            layers.Flatten(),
            layers.Dense(500, activation='sigmoid'),
            layers.Dense(8 * 8 * 4),
            layers.Reshape((8, 8, 4, 1))

        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv3DTranspose(256, (3, 3, 3), strides=(2, 2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Conv3DTranspose(128, (3, 3, 3), strides=(2, 2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Conv3DTranspose(1, (3, 3, 3), padding='same'),
            layers.Cropping3D(((14, 14, 7))),

        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

