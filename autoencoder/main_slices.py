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

from sklearn.model_selection import train_test_split

import preprocessing
import plotly.graph_objects as go
import matplotlib.pyplot as plt

path = 'C:/Users/agata/Documents/_POLSL/_IR3D/IR3D/data/CNN_100x100x50' #path = 'data/CNN_100x100x50'

dataset_interp_list = []
dataset_reconstr_list = []
for d in range(1, 28):
    model = f'd{d}'

    mat_interpolation = sio.loadmat(f'{path}/model/{model}.mat')
    mat_reconstruction = sio.loadmat(f'{path}/reconstruction_temp/{model}.mat')

    interpolation = mat_interpolation['Tintrp_mesh']
    reconstruction = mat_reconstruction['ir3d']

    reconstruction = ndimage.rotate(reconstruction, -90, reshape=False)

    interpolation[np.isnan(interpolation)] = 0
    interpolation = preprocessing.source_hole_filling(interpolation, 50)
    reconstruction = ndimage.rotate(reconstruction, -90)

    for angle in range(0, 30, 10):
        print(f"{model}: {angle}")
        interpolation_rotate = ndimage.rotate(interpolation, angle, reshape=False)
        reconstruction_rotate = ndimage.rotate(reconstruction, angle, reshape=False)

        dataset_interp_list.append(interpolation_rotate)
        dataset_reconstr_list.append(reconstruction_rotate)

dataset_interp = np.stack(dataset_interp_list, axis=0)
dataset_reconstr = np.stack(dataset_reconstr_list, axis=0)

# normalizatoin - model temperature cannot be lower than 20st,
# wokring on temperature difference, not absolute values
dataset_interp = dataset_interp - 20
dataset_interp[dataset_interp < 0] = 0

dataset_reconstr = dataset_reconstr - 20
dataset_reconstr[dataset_reconstr < 0] = 0

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(dataset_reconstr,
                                                            dataset_interp, test_size=0.2, random_state=0)

X_train = np.expand_dims(np.concatenate(X_train_v, axis=2).transpose(), axis=3)
X_test = np.expand_dims(np.concatenate(X_test_v, axis=2).transpose(), axis=3)
y_train = np.expand_dims(np.concatenate(y_train_v, axis=2).transpose(), axis=3)
y_test = np.expand_dims(np.concatenate(y_test_v, axis=2).transpose(), axis=3)

model = TemperatureAutoencoder((100, 100, 1))
model.compile(optimizer='adam', loss=losses.MeanSquaredError())
training_history = model.fit(X_train, y_train,
                             epochs=400,
                             batch_size=128,
                             shuffle=True,
                             validation_data=(X_test, y_test))

# # evaluation
test_case_ind = 5
test = np.expand_dims(X_test[test_case_ind], axis=0)
interp_test = y_test[test_case_ind]

encoded_img = model.encoder(test).numpy()
decoded_img = model.decoder(encoded_img).numpy()

fig, ax = plt.subplots(1, 3)
pos = ax[0].imshow(test[0, :, :, :], cmap='jet')
fig.colorbar(pos, ax=ax[0])

pos = ax[1].imshow(interp_test, cmap='jet')
fig.colorbar(pos, ax=ax[1])

pos = ax[2].imshow(decoded_img[0, :, :, :], cmap='jet')
fig.colorbar(pos, ax=ax[2])
fig.show()

# volume test
ind_test = 0
volume_test = X_test_v[ind_test]
volume_input = np.expand_dims(volume_test.transpose(), 3)
encoded_img = model.encoder(volume_input).numpy()
decoded_img = model.decoder(encoded_img).numpy()

volume_result = decoded_img[:, :, :, 0].transpose()
volume_plot.show(volume_result)