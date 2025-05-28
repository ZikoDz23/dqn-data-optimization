import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import layers 

def build_dueling_dqn(state_size, action_size):
    """
    Build a Dueling Deep Q-Network model
    """
    inputs = keras.Input(shape=(state_size,))
    
    # Shared hidden layers
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)

    # Dueling architecture
    # Value Stream
    value = layers.Dense(32, activation="relu")(x)
    value = layers.Dense(1)(value)

    # Advantage Stream
    advantage = layers.Dense(32, activation="relu")(x)
    advantage = layers.Dense(action_size)(advantage)

    # Combine value and advantage into Q-values
    q_values = layers.Lambda(lambda a: a[0] + (a[1] - tf.reduce_mean(a[1], axis=1, keepdims=True)))(
        [value, advantage]
    )

    model = keras.Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model
