
import numpy as np
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Input, Dense, Conv1D, MaxPooling1D,
                                     UpSampling1D, Flatten,Reshape,Cropping1D)
from useful import load_from_file, print_scores
from simple_autoencoder import tune_threshold, reconstruction, split_normal
tf.random.set_seed(100)
# # Define a CNN Autoencoder
def build_cnn_autoencoder(input_shape, learning_rate=0.01):
    inputs = Input(shape=input_shape)
    layer = inputs
    #Encoder
    for filter in [8,4]:
        layer = Conv1D(filter, kernel_size=3, activation='relu', padding='same')(layer)
        layer = MaxPooling1D(pool_size=2, padding='same')(layer)
        pool_shape=layer.shape[1:]
    #mlp layers
    layer=Flatten()(layer)
    flat_shape=layer.shape[1]
    layer=Dense(16,activation='relu')(layer)
    layer=Dense(flat_shape,activation='relu')(layer)
    layer=Reshape(pool_shape)(layer)
    # Decoder
    for filter in [4,8]:
        layer = Conv1D(filter, kernel_size=3, activation='relu', padding='same')(layer)
        layer = UpSampling1D(size=2)(layer)
    decoded = Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(layer)
    decoded = Cropping1D(cropping=(0, 3))(decoded)
    output = Reshape((input_shape[0],))(decoded)
    # Model
    autoencoder = Model(inputs, output)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder


if __name__ == "__main__":
    x,y = load_from_file(folder='binary',set="train")
    model = build_cnn_autoencoder((41,1),learning_rate=0.0005)
    model.summary()
    x_normal,x_anomaly = split_normal(x,y,0)
    model.fit(x_normal,x_normal,epochs=50,batch_size=10,validation_split=0.2)
    threshold = tune_threshold(model,x_anomaly)
    ##Prediction on test set
    print("\n\nPrediction on test set")
    reconstruction(model,set="test",threshold=threshold)


