
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import pickle
import numpy as np
import pandas as pd



try:
    with open('elasticnet_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

df = pd.DataFrame({
    'Date': pd.date_range(start='2024-06-03', periods=365)
})

df = pd.DataFrame({
    'Date': pd.date_range(start='2024-06-03', periods=365)  # Adjust as needed
})

def predict():
    try:
        # Check if the model was loaded
        if model is None:
            result_label.config(text="Model not loaded.")
            return

        # Collect the single feature from the input field
        feature = float(entry.get())

        # Reshape the feature to fit the model's expected input
        feature = np.array([feature]).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(feature)

        # Extract the predicted days value
        predicted_days = int(prediction[0]) if prediction.ndim == 1 else int(prediction[0, 0])

        # Calculate the predicted date
        min_date = df['Date'].min()  # Ensure this matches the Jupyter notebook DataFrame
        predicted_date = min_date + pd.to_timedelta(predicted_days, unit='D')
        formatted_date = predicted_date.strftime('%d-%b-%Y')

        # Display the prediction
        result_label.config(text=f"Predicted date for {int(feature[0, 0])} kg is {formatted_date}")

    except Exception as e:
        result_label.config(text=f"Error: {e}")
        print(f"Error during prediction: {e}")


# Create the main application window
root = tk.Tk()
root.title("ElasticNet Prediction")

# Create entry field for the single feature (weight)
label = tk.Label(root, text="Enter Weight")
label.pack()
entry = tk.Entry(root)
entry.pack()

# Create a button to trigger prediction
predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

# Start the application
root.mainloop()