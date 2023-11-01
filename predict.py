import tensorflow as tf
import numpy as np
import pandas as pd
import pywt
import argparse
import os

def process_csi(data):
    """
    Remove specific subcarrier columns from CSI data.

    Args:
        data: CSI data with subcarriers

    Returns:
        Processed CSI data with selected subcarriers removed.
    """
    # Define subcarriers to be removed
    subcarriers_to_remove = [x + 32 for x in [-21, -7, 21, 7, -32, -31, -30, -29, 31, 30, 29, 0]]
    subcarriers_to_remove.sort()
    # Remove the specified subcarriers
    return np.delete(data, subcarriers_to_remove, axis=1)

def mad(x):
    """
    Compute the Median Absolute Deviation of an array.

    Args:
        x: Input array or sequence

    Returns:
        Median Absolute Deviation of the array
    """
    return np.median(np.abs(x - np.median(x)))

def filter_hampel(signal, size=3, threshold=3):
    """
    Apply Hampel filter to identify and replace outliers in a signal.

    Args:
        signal: Input signal array
        size: Window size for the rolling operation
        threshold: Threshold for outlier detection

    Returns:
        Signal array after applying Hampel filter
    """
    signal_pd = pd.Series(signal.copy())
    scale = 1.4826
    # Calculate rolling median and MAD
    median_rolling = signal_pd.rolling(size, center=True).median().fillna(method='bfill').fillna(method='ffill')
    mad_rolling = scale * signal_pd.rolling(size, center=True).apply(mad).fillna(method='bfill').fillna(method='ffill')

    # Identify outliers
    outliers = np.where(np.abs(signal_pd - median_rolling) >= threshold * mad_rolling)[0]

    # Replace outliers with median values
    signal_pd[outliers] = median_rolling[outliers]
    return signal_pd.to_numpy()

def denoise_wavelet(signal, wavelet):
    """
    Apply Wavelet Denoising on the signal.

    Args:
        signal: Input signal array
        wavelet: Wavelet type for denoising

    Returns:
        Denoised signal
    """
    signal_copy = signal.copy()
    wavelet_function = pywt.Wavelet(wavelet)
    max_lvl = pywt.dwt_max_level(len(signal_copy), wavelet_function.dec_len)
    # Decompose and reconstruct signal using wavelets
    coeffs = pywt.wavedec(signal_copy, wavelet, max_lvl)
    reconstructed = pywt.waverec(coeffs, wavelet)
    return reconstructed

def prepare_input(input_data):
    """
    Preprocess input data for model prediction.

    Args:
        input_data: Raw input data with CSI and LGT components

    Returns:
        Preprocessed data ready for input into the model.
    """
    # Splitting CSI and LGT data
    csi_part = input_data[:, :64]
    lgt_part = input_data[:, 64:]
    # Process CSI and apply filters
    csi_preprocessed = process_csi(csi_part)
    csi_preprocessed = np.apply_along_axis(filter_hampel, 1, csi_preprocessed)
    csi_preprocessed = denoise_wavelet(csi_preprocessed, 'sym5')
    return [np.array([csi_preprocessed]), np.array([lgt_part])]

def predict_activity(data):
    """
    Predict human activity based on sensor data.

    Args:
        data: Preprocessed sensor data

    Returns:
        Predicted probabilities for each activity class.
    """
    preprocessed_data = prepare_input(data)
    # Load the trained model and predict
    trained_model = tf.keras.models.load_model('../model/v3_model')
    activity_probabilities = trained_model.predict(preprocessed_data)
    return activity_probabilities

def identify_activity(probabilities):
    """
    Identify the most likely activity from predicted probabilities.

    Args:
        probabilities: Array of predicted probabilities for each class

    Returns:
        The activity label with the highest probability.
    """
    activity_labels = ['empty', 'sit', 'stand', 'walk']
    return activity_labels[np.argmax(probabilities, axis=-1)[0]]

if __name__ == "__main__":
    # Setup command-line argument parsing
    arg_parser = argparse.ArgumentParser(description='Activity Prediction from Input Data')
    arg_parser.add_argument('-f', '--file', type=str, help='Path to input data file')
    arguments = arg_parser.parse_args()

    # Check for the existence of the input file
    if not os.path.isfile(arguments.file):
        raise FileNotFoundError(f"File not found: {arguments.file}")

    # Reading data from the specified file
    try:
        data_array = pd.read_csv(arguments.file, sep=",", header=None).values
    except Exception as err:
        raise Exception(f"Failed to read file {arguments.file}: {err}")
    
    # Predict activity based on the data
    predicted_probabilities = predict_activity(data_array)

    # Display the probabilities for each activity
    activity_classes = ['empty', 'sit', 'stand', 'walk']
    for i, cls in enumerate(activity_classes):
        print(f"{cls}: {predicted_probabilities[0][i]:.2%}")

    # Determine and display the most likely activity
    predicted_class = identify_activity(predicted_probabilities)
    print(f"File: {arguments.file}, Predicted Activity: {predicted_class}")
