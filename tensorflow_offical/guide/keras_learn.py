import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)
print(tf.keras.__version__)

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='sigmoid'))
model.add(layers.Dense(64, activation=tf.sigmoid))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
layers.Dense(64, kernle_initializer='orthogonal')
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation=tf.sigmoid),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

metrices=tf.keras.metrics.categorical_accuracy
gtf.keras.metrics.Accuracy
tf.keras.Input(shape=(32,))
