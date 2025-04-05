import tkinter as tk
from tkinter import messagebox
import numpy as np
from model import load_trained_model
from data_preprocessing import pad_and_reshape_input

# Load the model
model_path = "C:\College final project\stress_detection_model_90_percent.keras"  # Replace with your model's file path
model = load_trained_model(model_path)

def get_prediction(values):
    """
    Get the model prediction for the input values.
    Args:
        values (list): List of input values from the user.
    Returns:
        Prediction (stress probability).
    """
    try:
        input_data = np.array([values])
        preprocessed_data = pad_and_reshape_input(input_data, timesteps=4)
        prediction = model.predict(preprocessed_data)
        return prediction[0][0]  # Binary classification: output is in range [0, 1]
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make prediction: {e}")
        return None

# Create the interface
def create_interface():
    root = tk.Tk()
    root.title("Stress Detection Interface")
    root.geometry("400x400")

    # Input fields
    input_labels = ['bvp_mean', 'bvp_std', 'bvp_max', 'eda_mean', 'eda_std', 'eda_max', 
                    'temperature_mean', 'temperature_std', 'temperature_max']
    input_entries = {}

    for idx, label in enumerate(input_labels):
        tk.Label(root, text=label).grid(row=idx, column=0, padx=10, pady=5)
        entry = tk.Entry(root)
        entry.grid(row=idx, column=1, padx=10, pady=5)
        input_entries[label] = entry

    # Function for prediction
    def predict_stress():
        try:
            values = [float(input_entries[label].get()) for label in input_labels]
            prediction = get_prediction(values)
            if prediction is not None:
                stress_level = "Stressed" if prediction >= 0.5 else "Not Stressed"
                messagebox.showinfo("Prediction", f"Stress Level: {stress_level}\nProbability: {prediction:.2f}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric inputs.")

    # Predict button
    predict_button = tk.Button(root, text="Predict Stress Level", command=predict_stress)
    predict_button.grid(row=len(input_labels), column=0, columnspan=2, pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_interface()
