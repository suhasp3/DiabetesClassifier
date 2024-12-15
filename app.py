from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Safely extract form data - ensure the keys match index.html name attributes
        GenHlth = float(request.form['GenHlth'])
        HighBP  = float(request.form['HighBP'])
        BMI     = float(request.form['BMI'])
        DiffWalk = float(request.form['DiffWalk'])
        HighChol = float(request.form['HighChol'])
        Age     = float(request.form['Age'])

        # Put features into the correct order
        features = np.array([[GenHlth, HighBP, BMI, DiffWalk, HighChol, Age]])
        prediction = model.predict(features)[0]  # 0, 1, or 2
        prediction = int(prediction)

        return redirect(url_for('result', pred=prediction))

    # On GET request, render the form
    return render_template('index.html')

@app.route('/result/<pred>')
def result(pred):
    pred = int(pred)
    if pred == 0:
        prediction_text = "No Diabetes (0)"
    elif pred == 1:
        prediction_text = "Prediabetes (1)"
    else:
        prediction_text = "Diabetes (2)"

    # Very simple result page
    return f"""
    <div style='font-family: Arial, sans-serif; margin: 2em;'>
      <h2>Prediction: {prediction_text}</h2>
      <a href='/' style='text-decoration: none; color: #0d6efd;'>Go Back</a>
    </div>
    """

if __name__ == '__main__':
    app.run(debug=True)
