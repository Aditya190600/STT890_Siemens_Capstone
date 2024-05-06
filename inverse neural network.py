import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
import random

def random_split(y, test_size=0.20):
  #random.shuffle(y)
  # Split the data into training and test sets.
  n_test = int(test_size * len(y))
  n_test = 1000
  y_train = y[:n_test]
  y_test = y[n_test:]

  return y_train, y_test

# Example usage:


# Generate some random data for regression
np.random.seed(0)
X = np.random.rand(1500, 3)  # 1000 samples,
# X = np.linspace(0,1,1000)

# Setting up the blob parameters

# num_points = 900

# # Generate x and y coordinates
# x = np.linspace(0, 1, int(np.sqrt(num_points)))
# y = np.linspace(0, 1, int(np.sqrt(num_points)))

# # Create meshgrid
# X, Y = np.meshgrid(x, y)

# # Reshape to get a 1D array of points
# points = np.vstack([X.ravel(), Y.ravel()]).T

# # Generate x and y coordinates
# x_100= np.linspace(0, 1, int(np.sqrt(100)))
# y_100 = np.linspace(0, 1, int(np.sqrt(100)))

# # Create meshgrid
# X, Y = np.meshgrid(x_100, y_100)

# # Reshape to get a 1D array of points
# test_points = np.vstack([X.ravel(), Y.ravel()]).T

n_samples = 1500
n_features = 2
centers = [ (0, 0), (5, 5)]
y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)[0]
# y = make_moons(noise=0.05, n_samples=n_samples)[0]
# Split the data into training and testing sets
X_train, X_test = random_split(X)
y_train, y_test = random_split(y)


def custom_loss(y_true, y_pred):
    first_expanded = tf.expand_dims(y_true, axis=1)
    second_expanded = tf.expand_dims(y_pred, axis=0)
    distances = tf.reduce_sum(tf.square(first_expanded - second_expanded), axis=2)
    closest_indices = tf.reduce_min(distances, axis=1)
    closest_indices_1 = tf.reduce_min(distances, axis=0)
    average_closest_points = tf.reduce_sum(closest_indices, axis=0)
    return tf.math.log(average_closest_points)


# Define the neural network architecture
## add bias 

model = Sequential([
    Dense(32, activation='relu', input_shape=(3,),bias_initializer='zeros'),
    Dense(64, activation='relu',bias_initializer='zeros'),
    # Dropout(0.5),
    Dense(128, activation='relu',bias_initializer='zeros'),
    Dense(256, activation='relu',bias_initializer='zeros'),
    Dense(512, activation='relu',bias_initializer='zeros'),
    Dense(256, activation='relu',bias_initializer='zeros'),
    Dense(64, activation='relu',bias_initializer='zeros'),
    Dense(2)  # Output layer with two neurons for regression
])

# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss)

# Custom callback for plotting results after every 100 epochs
class PlotResults(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            predictions = self.model.predict(X_test)
            plt.scatter(y_test[:, 0], y_test[:, 1], color='blue', label='Original Data')
            plt.scatter(predictions[:, 0], predictions[:, 1], color='red', label='Predictions')
            plt.title(f'Regression with Custom Loss (Epoch {epoch})')
            plt.xlabel('Y1')
            plt.ylabel('Y2')
            plt.legend()
            plt.savefig(f"C:/Users/kevin/OneDrive/Documents/MSU/Sem 4/Capstone/3_var_2_blob/epoch {epoch}.png",dpi=300)
            plt.close()

# Train the model with custom callback
history = model.fit(X_train, y_train, epochs=1200, batch_size=1000, validation_split=0.2, callbacks=[PlotResults()])
# pred= model.predict(X)
# plt.scatter(y[:, 0], y[:, 1], color='green', label='Original Data')
# plt.scatter(pred[:, 0], pred[:, 1], color='blue', label='Predictions')
# plt.xlabel('Y1')
# plt.ylabel('Y2')
# plt.legend()
# plt.show()
# plt.close()


# plt.semilogy(history.history['loss'],label='Training Loss')
# plt.semilogy(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)


