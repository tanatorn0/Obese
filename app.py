from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("Obese_1.pkl")

# Mapping of prediction results
obesity_categories = {
    0: 'ผอม',
    1: 'ปกติ',
    2: 'นํ้าหนักเกิน',
    3: 'อ้วน'
}

@app.route('/api/Obese', methods=['POST'])
def house():
    # Get the values from the form with default values in case of None
    age = request.form.get('age', '0')  # Default to '0' if not provided
    Gender = request.form.get('Gender', '0')  # Default to '0' if not provided
    Height = request.form.get('Height', '0.0')  # Default to '0.0' if not provided
    Weight = request.form.get('Weight', '0.0')  # Default to '0.0' if not provided

    # Convert the values to appropriate types
    try:
        age = int(age)
        Gender = int(Gender)
        Height = float(Height)
        Weight = float(Weight)
    except ValueError:
        return {'error': 'Invalid input'}, 400

    # Prepare the input for the model
    x = np.array([[age, Gender, Height, Weight]])

    # Predict using the model
    prediction = model.predict(x)

    # Return the result with the corresponding category
    result_category = obesity_categories.get(int(prediction[0]), 'Unknown')
    return {'ObesityCategory': result_category}, 200    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
