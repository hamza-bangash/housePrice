from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the pre-trained model
with open('ridge_model.pkl', 'rb') as f:
    model =pickle.load(f)
# Load the column types and categories
with open('col_types.pkl', 'rb') as f:
    col_types = pickle.load(f)
# Load the categories for categorical features
with open('categories.pkl', 'rb') as f:
    categories = pickle.load(f)
# Define the feature names based on the column types
FEATURE_NAMES = list(col_types.keys())

@app.route('/')
def index():
    return  render_template(
        'index.html',
        col_types=col_types,
        categories=categories
    )



@app.route('/predict', methods=['POST'])
def predict():
    data = {}
    for col in FEATURE_NAMES:
        val = request.form[col]
        dtype = col_types[col]
        if 'float' in dtype:
            data[col] = float(val)
        elif 'int' in dtype:
            data[col] = int(val)
        else:
            data[col] = val
    df = pd.DataFrame([data], columns=FEATURE_NAMES)
    price = model.predict(df)[0]
    price = price**2  # Square the prediction to get the final price
    price = int(price)  # Convert to integer for display
    return render_template('result.html', price=price)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({"prediction": int(prediction**2)})

if __name__ == "__main__":
    app.run(debug=True)

