import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import layers 
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.layers import Input, Dense, Lambda, Add, Subtract, BatchNormalization, Dropout
from keras._tf_keras.keras.optimizers import Adam
import tensorflow as tf

def build_dueling_dqn(input_dim, action_size):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)

    # Dueling streams
    value_fc = Dense(64, activation='relu')(x)
    value = Dense(1)(value_fc)

    advantage_fc = Dense(64, activation='relu')(x)
    advantage = Dense(action_size)(advantage_fc)

    advantage_mean = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    q_values = Add()([value, Subtract()([advantage, advantage_mean])])

    model = Model(inputs=inputs, outputs=q_values)

    # Gradient clipping
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='huber')  # plus robuste que MSE

    return model
