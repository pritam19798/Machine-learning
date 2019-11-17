import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


training_data=np.load('my_data.npz')
input_size=2
output_size=1


model=tf.keras.Sequential([tf.keras.layers.Dense(output_size)])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(training_data['inputs'],training_data['target'],epochs=100,verbose=2)
print(model.layers[0].get_weights())


plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])),np.squeeze(training_data['target']))
plt.xlabel('output')
plt.ylabel('target')
plt.show()
