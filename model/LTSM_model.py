import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump
from matplotlib.pyplot import figure
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Directory paths
current_directory = os.getcwd()
data_directory = r"..\dataset\230510_combined_data"

# Define constants for data processing
PILOT_SUBCARRIERS = ['csi_channel_' + str(i+32) for i in [-21, -7, 21,  7]]
NULL_SUBCARRIERS = ['csi_channel_' + str(i+32) for i in [-32, -31, -30, -29, 31,  30,  29,  0]]
SUBCARRIERS_TO_REMOVE = PILOT_SUBCARRIERS + NULL_SUBCARRIERS

# Function to load CSV data
def load_csv_data(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            label = filename.split("_")[0]
            yield filepath, label

# Load data from directory
activity_files = list(load_csv_data(data_directory))

# Functions for data processing
def process_data(file_paths):
    data = []
    for path in file_paths:
        df = pd.read_csv(path, header=None)
        data.append(df.values)
    return data

def assign_labels(data, label):
    labels = [label for _ in range(len(data))]
    return np.array(labels).reshape(-1, 1)

# Data processing
categories = ['empty', 'stand', 'sit', 'walk']
file_paths = {cat: [f for f, lbl in activity_files if lbl == cat] for cat in categories}

data = {cat: process_data(file_paths[cat]) for cat in categories}
labels = {cat: assign_labels(data[cat], cat) for cat in categories}

# Data concatenation and shuffling
X_all = np.vstack([data[cat] for cat in categories])
y_all = np.vstack([labels[cat] for cat in categories])
X_all, y_all = shuffle(X_all, y_all, random_state=5)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=5)

# Scaling features
class DataScaler(StandardScaler):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

# Apply scaling
scaler = DataScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encode labels
encoder = OneHotEncoder()
y_train_encoded = encoder.fit_transform(y_train).toarray()
y_test_encoded = encoder.transform(y_test).toarray()

# Save scaler and encoder
os.makedirs('v6_obj', exist_ok=True)
dump(scaler, open('v6_obj/scaler.pkl', 'wb'))
dump(encoder, open('v6_obj/encoder.pkl', 'wb'))

# Defining LSTM Architecture
lstm_net = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(48, input_shape=X_train[-2,:].shape),  # Adjusted input_shape
    tf.keras.layers.Dropout(0.1),  # Dropout layer
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(y_train_encoded.shape[1], activation='softmax')
])

# Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    return 1e-8 * 10**(epoch / 20)

schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# Optimizer Configuration
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
lstm_net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training the Model
history = lstm_net.fit(X_train, y_train_encoded, epochs=150, validation_split=0.2, batch_size=16, callbacks=[schedule])

# Visualizing Learning Rate Impact
plt.figure(figsize=(12, 8))
plt.semilogx(history.history["lr"], history.history["loss"])
sns.set_style("whitegrid")
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate vs Loss')

# Callback for Early Stopping
class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.98:
            print("\nReached 98% accuracy. Stopping training.")
            self.model.stop_training = True

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
my_callback = TrainingCallback()

# LSTM Network Reinitialization for Training with Optimal Learning Rate
final_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(48, input_shape=X_train[-2,:].shape),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.Dense(y_train_encoded.shape[1], activation='softmax')
])

# Model Compilation and Training
final_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])
final_history = final_model.fit(X_train, y_train_encoded, epochs=500, validation_split=0.2, batch_size=16, callbacks=[my_callback, early_stop])

# Model Evaluation on Test Data
final_model.evaluate(X_test, y_test_encoded)

# Plotting Training History
plt.figure(figsize=(10, 5))
plt.plot(final_history.history['accuracy'], label='Training Accuracy')
plt.plot(final_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Confusion Matrix and Classification Report
y_pred = final_model.predict(X_test)
conf_matrix = confusion_matrix(np.argmax(y_test_encoded, axis=1), np.argmax(y_pred, axis=1))
activity_labels = ['Empty', 'Sit', 'Stand', 'Walk']

conf_df = pd.DataFrame(conf_matrix, index=activity_labels, columns=activity_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_df, annot=True, fmt='g', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Displaying Classification Report
print(classification_report(np.argmax(y_test_encoded, axis=1), np.argmax(y_pred, axis=1), target_names=activity_labels))

# Saving the Final Model
final_model.save('final_activity_model.h5')
