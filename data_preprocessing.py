import numpy as np

def pad_and_reshape_input(data, timesteps=4):
    """
    Pads and reshapes the input data for model inference.
    Args:
        data (numpy.ndarray): Input data for preprocessing.
        timesteps (int): Number of timesteps for reshaping.
    Returns:
        Preprocessed data.
    """
    padding_needed = timesteps - data.shape[1] % timesteps
    if padding_needed != 0:
        padding = np.zeros((data.shape[0], padding_needed))
        data_padded = np.concatenate([data, padding], axis=1)
    else:
        data_padded = data

    return data_padded.reshape(data_padded.shape[0], timesteps, -1)
