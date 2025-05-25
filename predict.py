# predict.py

import numpy as np
import joblib

# Load the saved model
model = joblib.load('model.pkl')

# Predict function
def predict_object(input_data):
    try:
        # Convert input string of numbers into numpy array
        input_array = np.asarray(input_data, dtype=float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)

        # Return readable result
        if prediction[0] == 'R':
            return "This object is a Rock"
        else:
            return "This object is a Mine"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
