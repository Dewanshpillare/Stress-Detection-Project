import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to load the trained model
def load_trained_model(model_path):
    """
    Loads the trained model from the specified path.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        Loaded Keras model.
    """
    model = load_model(model_path)
    return model
