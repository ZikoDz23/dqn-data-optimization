import tensorflow as tf
import keras._tf_keras
from keras._tf_keras.keras.layers import Layer

class ReduceMeanLayer(Layer):
    """
    Couche personnalisée pour encapsuler tf.reduce_mean.
    """
    def __init__(self, axis=None, keepdims=False):
        super(ReduceMeanLayer, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):
        """
        Applique tf.reduce_mean sur les inputs.
        """
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)

class DuelingLayer(Layer):
    """
    Couche personnalisée pour Dueling DQN.
    """
    def __init__(self):
        super(DuelingLayer, self).__init__()
        self.reduce_mean_layer = ReduceMeanLayer(axis=1, keepdims=True)

    def call(self, inputs):
        """
        Implémente la logique de la couche Dueling DQN.
        """
        value, advantage = inputs
        mean_advantage = self.reduce_mean_layer(advantage)
        return value + (advantage - mean_advantage)

def build_dueling_dqn(state_size, action_size):
    """
    Construit un modèle Dueling DQN.
    """
    inputs = keras._tf_keras.keras.Input(shape=(state_size,))
    x = keras._tf_keras.keras.layers.Dense(128, activation='relu')(inputs)
    x = keras._tf_keras.keras.layers.Dense(64, activation='relu')(x)
    value = keras._tf_keras.keras.layers.Dense(1, activation='linear')(x)
    advantage = keras._tf_keras.keras.layers.Dense(action_size, activation='linear')(x)
    q_values = DuelingLayer()([value, advantage])
    model = keras._tf_keras.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=keras._tf_keras.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model