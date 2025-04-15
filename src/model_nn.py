from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Normalization

class ModelNN:
    
    def __init__(self, n_hidden_layers = 10, n_neurons = 128):
        """Initialize a standard sequential neural network with tensorflow.
        Parameters
        ------------
        n_hidden_layers: int
            Number of hidden layers.
        n_neurons: int
            Number of neurons to place in the largest of the hidden layers.
        """
        