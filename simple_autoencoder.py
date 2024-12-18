from network_class import NeuralNetworkManager
from useful import load_from_file, reconstruction, split_normal
import numpy as np
import tensorflow as tf
tf.random.set_seed(300)


if __name__ == "__main__":
    x,y = load_from_file(folder='binary',set="train")

    #separate the training set into normal and anamoly
    x_normal,x_anomaly = split_normal(x,y,0)
    autoE = NeuralNetworkManager("simple_autoencoder", x_train_ndim=x.shape[1],folder='autoE',learning_rate=0.001)
    autoE.create_network([32,16,8,16,32],output_num=x.shape[1],output_actiavtion='linear')
    autoE.compile_model(loss='mse',eval_metric='mse')

    #fit the model to reconstruct the normal database
    autoE.fit_model(x_normal,x_normal,epochs=20,batch_size=10)

    #calculate the loss on the anomalies to set the threshold
    anamoly_loss = np.abs(autoE.model.predict(x_anomaly) - x_anomaly)
    threshold = np.min(anamoly_loss) +  0.3*np.std(anamoly_loss) #tune for better performance

    print("Threshold = ",threshold)
    if threshold < 0:raise ValueError("too small")
    if threshold >np.mean(anamoly_loss):raise ValueError("too big")
    #reconstruct entire train set
    reconstruction(autoE.model,set="train",threshold=threshold) #tune threshold value inaccordance to the confusion matrix


    #With the new threshold predict on the test set
    print("Predicting on the test set")
    reconstruction(autoE.model,set="test",threshold=threshold)

