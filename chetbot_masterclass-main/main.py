import numpy as np
import tensorflow as tf
from keras import layers
from data_preprocess import X_train, y_train

# The chatbot brain creation : It's an artificial neural network of 3 layers
input_size = X_train.shape[1]
output_size = np.unique(y_train).shape[0]
chatbot_brain = tf.keras.Sequential(
    [
        layers.Dense(100, input_shape=(input_size,), activation='relu', name='layer1'),
        layers.Dense(50, activation='relu', name='layer2'),
        layers.Dense(output_size, activation='sigmoid', name='layer3')
    ]
)
chatbot_brain.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.008)
)

# train the chatbot brain
history = chatbot_brain.fit(
    X_train,
    y_train,
    epochs= 70,
    batch_size= 64,
    verbose = 1,
)

# save the chatbot_brain into a file called 'chatbot_brain.keras'
chatbot_brain.save('Barista.keras')