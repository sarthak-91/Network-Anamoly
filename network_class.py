import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, History
class NeuralNetworkManager:
    def __init__(self, model_name, x_train_ndim=None, folder="smaller", learning_rate=1e-4):
        """
        Initialize the Neural Network Manager.
        
        Args:
            data (Dataset): Dataset object containing training and testing data
            model_name (str): Name of the model
            folder (str, optional): Folder to save models. Defaults to "smaller".
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        """
        self.x_train_ndim = x_train_ndim
        self.folder = folder
        self.learning_rate = learning_rate
        self.model = None
        self.trained = False
        self.model_name = model_name
        self.optimizer = Adam(learning_rate=learning_rate)
        self.compiled = False
        self.history = None
        self.ensure_folder_exists(self.folder)
    def ensure_folder_exists(self, path):
        """
        Ensure the specified directory exists.
        
        Args:
            path (str): Directory path to create
        """
        os.makedirs(path, exist_ok=True)

    def create_network(self, dense_list,output_num=None, output_actiavtion=None,model_name=None):
        """
        Create neural network model.
        
        Args:
            dense_list (list): List of neurons for hidden layers
            model_name (str, optional): Name of the model. Defaults to None.
        
        Raises:
            ValueError: If input data is invalid or dense_list is not provided
        """
        if self.x_train_ndim is None:
            raise ValueError("Invalid input data shape for model creation.")
        
        if not dense_list:
            raise ValueError("Please specify a list of neurons for hidden layers. "
                             "E.g., [10, 10] for two layers with 10 neurons each.")
        
        if model_name: self.model_name = model_name
        input_layer = Input(shape=(self.x_train_ndim,))
        x = input_layer
        #x = Flatten()(input_layer)
        for units in dense_list:
            x = Dense(units)(x)
            x = LeakyReLU(negative_slope=0.3)(x)
        
        if output_actiavtion is None: act = 'linear'
        else: act = output_actiavtion
        if output_num is None:
            output_layer = Dense(1,activation=act)(x)
        else: output_layer = Dense(output_num,activation=act)(x)
        self.model = Model(input_layer, output_layer, name=model_name)
        print(f"Model '{self.model_name}' created.")

    def compile_model(self, summarize=False, save=False, optimizer=None,loss=None,eval_metric=None):
        """
        Compile the neural network model.
        
        Args:
            summarize (bool, optional): Print model summary. Defaults to False.
            save (bool, optional): Save compiled model. Defaults to False.
        
        Raises:
            ValueError: If no model has been defined
        """
        if self.model is None:
            raise ValueError("No model has been defined. Use `create_network` first.")
        loss_metric='mse' if loss is None else loss
        eval_metric='mse' if eval_metric is None else eval_metric
        if optimizer != None: self.optimizer = optimizer
        self.model.compile(optimizer=self.optimizer, loss=loss_metric, metrics=[eval_metric])
        self.compiled = True
        
        if summarize:
            self.model.summary()

        if save:
            dir_name = os.path.join(self.folder, self.model_name)
            self.ensure_folder_exists(dir_name)
            self.model.save(os.path.join(dir_name, f"{self.model_name}.keras"))

    def fit_model(self, x_train,y_train,epochs=250, save_checkpoint=False, save_model=True, batch_size=32):
        """
        Train the neural network model.
        
        Args:
            epochs (int, optional): Number of training epochs. Defaults to 250.
            save_checkpoint (bool, optional): Save model checkpoints. Defaults to False.
            save_model (bool, optional): Save final model. Defaults to True.
            batch_size (int, optional): Training batch size. Defaults to 32.
        
        Raises:
            ValueError: If model is not compiled or created
        """
        if not self.compiled or self.model is None:
            raise ValueError("Model must be created and compiled before training.")

        if x_train.shape[1] != self.x_train_ndim:
            raise ValueError("Specified shapes dont match")
        callbacks = []
        if save_checkpoint:
            checkpoint_path = os.path.join(self.folder, self.model_name, 
                                           "cp-{epoch:04d}.ckpt")
            self.ensure_folder_exists(os.path.dirname(checkpoint_path))
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path, 
                save_weights_only=True, 
                save_freq="epoch", 
                verbose=1
            ))

        self.history = self.model.fit(
            x_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True, 
            validation_split=0.2, 
            callbacks=callbacks
        )
        self.trained = True

        if save_model:
            save_path = os.path.join(
                self.folder, 
                f"{self.model_name}_{epochs}.keras"
            )
            self.model.save(save_path)
            
            # Save training history
            history_path = os.path.join(
                self.folder,  
                f"{self.model_name}_{epochs}_history.json"
            )
            json.dump(self.history.history, open(history_path, 'w'))

    def predict(self,x_test,y_test):
        """
        Make predictions on test data.
        
        Returns:
            tuple: Predictions and Mean Squared Error
        
        Raises:
            AttributeError: If model is not ready for prediction
        """
        if not self.trained or not self.compiled:
            raise AttributeError("Model must be trained and compiled before prediction.")
        print(y_test.shape)
        predictions = self.model.predict(x_test)
        print(predictions.shape)
        mse = (np.abs(y_test- predictions)).mean()
        
        print(f"Mean Squared Error: {mse}")
        return predictions, mse

    def load_model(self, model_name, epochs):
        """
        Load a previously saved model.
        
        Args:
            model_name (str): Name of the model to load
            epochs (int): Number of epochs in the saved model
        
        Raises:
            FileNotFoundError: If specified model does not exist on path 
        """
        path = os.path.join(self.folder, f"{model_name}_{epochs}.keras")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = load_model(path)
        self.model_name = model_name
        self.compiled = True
        self.trained = True
