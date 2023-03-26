import tensorflow as tf
import os


def get_model(last_days = 30) -> tf.keras.models.Sequential:
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape= (last_days-1, 1)))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dense(25))
    model.add(tf.keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    checkpoint_path = "training_1/cp.ckpt"
    model.load_weights(checkpoint_path)

    return model

