from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data from the request
    data = request.json

    # Extract features from the input
    try:
        # Example: Ensure these match your dataset feature names
        features = [
            data['HighBP'], data['HighChol'], data['BMI'], 
            data['GenHlth'], data['DiffWalk'], data['Age']
        ]
        # Convert to a numpy array for the model
        features = np.array(features).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)[0]  # Output will be 0, 1, or 2

        # Map prediction to a human-readable result if needed
        diabetes_classes = {
            0: "No Diabetes",
            1: "Prediabetes",
            2: "Diabetes"
        }
        result = diabetes_classes[prediction]

        # Return the result as JSON
        return jsonify({"prediction": result})

    except KeyError as e:
        # Handle missing features
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
