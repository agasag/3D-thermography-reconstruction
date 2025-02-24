import numpy as np
import scipy.io as sio

import volume_plot
from scipy import ndimage, misc
from sklearn.preprocessing import MinMaxScaler

from temp_autoencoder import TemperatureAutoencoder
from temp_autoencoder import TemperatureAutoencoder3D
from temp_autoencoder import TemperatureAutoencoderMix
from temp_autoencoder import TemperatureAutoencoderAS # conv3d

from getConn import getConn # CNN with layers connection (weights transfer), conv3d

from tensorflow.keras import losses
import tensorflow as tf

from sklearn.model_selection import train_test_split

import preprocessing
import plotly.graph_objects as go
import glob
import os

path_files = 'C:/Users/agata/Documents/_POLSL/_IR3D/_IR3D/IR3D_Branch_DL/IR3D/autoencoder/CNN_100x100x50/_model_mat'
path = 'C:/Users/agata/Documents/_POLSL/_IR3D/_IR3D/IR3D_Branch_DL/IR3D/autoencoder/CNN_100x100x50'

is_conv3d = False

dataset_interp = []
dataset_reconstr = []

files = [f for f in glob.glob(path_files + "**/*.mat", recursive=True)]

# for d in range(1, 10):
for file in files:
    # model = f'd{d}'
    model = os.path.basename(file)

    mat_interpolation = sio.loadmat(f'{path}/_model_mat/{model}')
    mat_reconstruction = sio.loadmat(f'{path}/_reco_mat/{model}')

    interpolation = mat_interpolation['Tintrp_mesh']
    reconstruction = mat_reconstruction['ir3d']

    interpolation[np.isnan(interpolation)] = 0
    interpolation = preprocessing.source_hole_filling(interpolation, 50)

    # volume_plot.show(interpolation)
    # volume_plot.show(reconstruction)
    # TODO: input pipelines to use every angle
    for angle in range(0, 360, 10):
        print(f"{model}: {angle}")
        interpolation_rotate = ndimage.rotate(interpolation, angle, reshape=False)
        reconstruction_rotate = ndimage.rotate(reconstruction, angle, reshape=False)

        dataset_interp.append(interpolation_rotate)
        dataset_reconstr.append(reconstruction_rotate)

dataset_interp = np.stack(dataset_interp, axis=0)
dataset_reconstr = np.stack(dataset_reconstr, axis=0)

# normalizatoin - model temperature cannot be lower than 20st,
# wokring on temperature difference, not absolute values
dataset_interp = dataset_interp - 20
dataset_interp[dataset_interp < 0] = 0

dataset_reconstr = dataset_reconstr - 20
dataset_reconstr[dataset_reconstr < 0] = 0

# for conv3D
if is_conv3d:
    dataset_interp = np.expand_dims(dataset_interp, axis=4)
    dataset_reconstr = np.expand_dims(dataset_reconstr, axis=4)

X_train, X_test, y_train, y_test = train_test_split(dataset_reconstr,
                                                    dataset_interp, test_size=0.1, random_state=0)

# model = TemperatureAutoencoder(interpolation.shape)
# model.compile(optimizer='adam', loss=losses.MeanSquaredError())
# model.fit(X_train, y_train, epochs=100, shuffle=True,
#           validation_data=(X_test, y_test))

# model = TemperatureAutoencoder3D((100, 100, 50, 1))
model = TemperatureAutoencoder((100, 100, 50))
model.compile(optimizer='adam', loss=losses.MeanSquaredError())
training_history = model.fit(X_train, y_train,
                             epochs=500,
                             batch_size=64,
                             shuffle=True,
                             validation_data=(X_test, y_test))

test_case_ind = 5
test = np.expand_dims(X_test[test_case_ind], axis=0)
interp_test = y_test[test_case_ind]

encoded_img = model.encoder(test).numpy()
decoded_img = model.decoder(encoded_img).numpy()
decoded_img = decoded_img[0]

if is_conv3d:
    test_ex = interp_test[:, :, :, 0]
    output_ex = decoded_img[:, :, :, 0]
else:
    test_ex = interp_test
    output_ex = decoded_img

volume_plot.show(test_ex)
volume_plot.show(output_ex)

difference = test_ex - output_ex
volume_plot.show(difference)

fig = go.Figure()
fig.add_trace(go.Scatter(y=training_history.history["loss"], name="loss"))
fig.add_trace(go.Scatter(y=training_history.history["val_loss"], name="val_loss"))
fig.show()