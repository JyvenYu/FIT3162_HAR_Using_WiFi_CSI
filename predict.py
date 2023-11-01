import tensorflow as tf
import numpy as np
import pandas as pd
import pywt
import argparse
import os

def process_csi(data):
    subcarriers_to_remove = [x + 32 for x in [-21, -7, 21, 7, -32, -31, -30, -29, 31, 30, 29, 0]]
    subcarriers_to_remove.sort()
    return np.delete(data, subcarriers_to_remove, axis=1)

def mad(x):
    """
    Compute Median Absolute Deviation
    """
    return np.median(np.abs(x - np.median(x)))

def filter_hampel(signal, size=3, threshold=3):
    """
    Apply Hampel Filter
    """
    signal_pd = pd.Series(signal.copy())
    scale = 1.4826
    median_rolling = signal_pd.rolling(size, center=True).median().fillna(method='bfill').fillna(method='ffill')
    mad_rolling = scale * signal_pd.rolling(size, center=True).apply(mad).fillna(method='bfill').fillna(method='ffill')

    outliers = np.where(np.abs(signal_pd - median_rolling) >= threshold * mad_rolling)[0]

    signal_pd[outliers] = median_rolling[outliers]
    return signal_pd.to_numpy()

def denoise_wavelet(signal, wavelet):
    signal_copy = signal.copy()
    wavelet_function = pywt.Wavelet(wavelet)
    max_lvl = pywt.dwt_max_level(len(signal_copy), wavelet_function.dec_len)
    coeffs = pywt.wavedec(signal_copy, wavelet, max_lvl)
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed

def prepare_input(input_data):
    csi_part = input_data[:, :64]
    lgt_part = input_data[:, 64:]
    csi_preprocessed = process_csi(csi_part)
    csi_preprocessed = np.apply_along_axis(filter_hampel, 1, csi_preprocessed)
    csi_preprocessed = denoise_wavelet(csi_preprocessed, 'sym5')
    return [np.array([csi_preprocessed]), np.array([lgt_part])]

def predict_activity(data):
    """ Predict activity from sensor data """
    preprocessed_data = prepare_input(data)

    # Loading pre-trained model
    trained_model = tf.keras.models.load_model('../model/v3_model')

    # Predicting using the model
    activity_probabilities = trained_model.predict(preprocessed_data)
    return activity_probabilities

def identify_activity(probabilities):
    """ Identify most likely activity """
    activity_labels = ['empty', 'sit', 'stand', 'walk']
    return activity_labels[np.argmax(probabilities, axis=-1)[0]]

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description='Activity Prediction from Input Data')
    arg_parser.add_argument('-f', '--file', type=str, help='Path to input data file')
    arguments = arg_parser.parse_args()

    # Verify if file exists
    if not os.path.isfile(arguments.file):
        raise FileNotFoundError(f"File not found: {arguments.file}")

    # Reading data from the file
    try:
        data_array = pd.read_csv(arguments.file, sep=",", header=None).values
    except Exception as err:
        raise Exception(f"Failed to read file {arguments.file}: {err}")
    
    predicted_probabilities = predict_activity(data_array)
    activity_classes = ['empty', 'sit', 'stand', 'walk']
    for i, cls in enumerate(activity_classes):
        print(f"{cls}: {predicted_probabilities[0][i]:.2%}")

    predicted_class = identify_activity(predicted_probabilities)
    print(f"File: {arguments.file}, Predicted Activity: {predicted_class}")
