from flask import Flask, request, render_template
import joblib
import numpy as np

# For data visualization
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json

# For dtreeviz
from dtreeviz import dtreeviz
import os

from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load your saved model
model = joblib.load('diabetes_model.pkl')

# Mock population averages for a basic bar chart:
POPULATION_AVERAGES = {
    'GenHlth': 3.0,   # "average" general health rating
    'BMI': 28.0,      # average BMI
    'Age': 45.0,      # average age
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    user_data = {}
    feature_importance = {}
    dtree_html = None

    if request.method == 'POST':
        # Extract form data
        GenHlth  = float(request.form['GenHlth'])
        HighBP   = float(request.form['HighBP'])
        BMI      = float(request.form['BMI'])
        DiffWalk = float(request.form['DiffWalk'])
        HighChol = float(request.form['HighChol'])
        Age      = float(request.form['Age'])

        user_data = {
            'GenHlth': GenHlth,
            'HighBP': HighBP,
            'BMI': BMI,
            'DiffWalk': DiffWalk,
            'HighChol': HighChol,
            'Age': Age
        }

        # Prediction
        features = np.array([[GenHlth, HighBP, BMI, DiffWalk, HighChol, Age]])
        pred = model.predict(features)[0]  # could be 0,1,2
        prediction = int(pred)

        # Feature importance (if the model has .feature_importances_)
        # We'll assume the same order: [GenHlth, HighBP, BMI, DiffWalk, HighChol, Age]
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
            columns = ['GenHlth', 'HighBP', 'BMI', 'DiffWalk', 'HighChol', 'Age']
            feature_importance = dict(zip(columns, importance_values))

        # Generate decision tree visualization (HTML) with dtreeviz
        try:
            # dtreeviz requires the original data and target to build the visuals
            # For demonstration, let's just re-train the same model on some small subset of data.
            # If you still have X_train, y_train, you could load them. Or use the balanced data snippet.
            # If not, we'll just do a minimal approach to generate a small tree viz.

            # NOTE: This is just an illustration. If you still have the training data X_train, y_train:
            # from joblib import load
            # X_train, y_train = load('X_train.pkl'), load('y_train.pkl')
            # Then generate viz with those actual data.

            # For demonstration, let's assume the model is a DecisionTreeClassifier:
            if isinstance(model, DecisionTreeClassifier):
                # We'll attempt a dtreeviz with random/placeholder data
                # A small artificial dataset:
                # dtreeviz must know the feature names and target names
                dummy_X = np.array([
                    [2, 1, 27, 0, 1, 35],
                    [3, 0, 24, 0, 0, 22],
                    [5, 1, 40, 1, 1, 55],
                    [1, 0, 18, 0, 0, 30],
                    [4, 1, 30, 1, 1, 50],
                ])
                dummy_y = np.array([2,0,2,0,1])  # some random classes

                # Re-fit a small DecisionTree just for dtreeviz demonstration:
                temp_tree = DecisionTreeClassifier(max_depth=3)
                temp_tree.fit(dummy_X, dummy_y)

                viz = dtreeviz(
                    temp_tree,
                    dummy_X,
                    dummy_y,
                    target_name="Diabetes_012",
                    feature_names=["GenHlth","HighBP","BMI","DiffWalk","HighChol","Age"],
                    class_names=["No Diabetes(0)", "Pre(1)", "Yes Diabetes(2)"]  # for dtreeviz
                )
                dtree_html = viz._repr_svg_()  # dtreeviz returns an SVG embed
            else:
                dtree_html = "<p>Interactive tree visualization only supported for DecisionTreeClassifier.</p>"
        except Exception as e:
            dtree_html = f"<p>Error generating dtreeviz: {str(e)}</p>"

    # Build Plotly figure comparing user’s input vs population average
    comparison_chart = None
    if user_data:  # Means we have POST data
        fig = go.Figure()
        features_to_compare = ['GenHlth','BMI','Age']
        user_vals = [user_data[f] for f in features_to_compare]
        pop_vals = [POPULATION_AVERAGES[f] for f in features_to_compare]

        fig.add_trace(go.Bar(
            x=features_to_compare,
            y=user_vals,
            name='User Input',
            marker_color='indianred'
        ))
        fig.add_trace(go.Bar(
            x=features_to_compare,
            y=pop_vals,
            name='Population Avg',
            marker_color='lightseagreen'
        ))
        fig.update_layout(
            title="Your Input vs. Population Averages",
            barmode='group'
        )
        comparison_chart = json.dumps(fig, cls=PlotlyJSONEncoder)

    return render_template(
        'index.html',
        prediction=prediction,
        user_data=user_data,
        feature_importance=feature_importance,
        comparison_chart=comparison_chart,
        dtree_html=dtree_html
    )

if __name__ == '__main__':
    app.run(debug=True)
