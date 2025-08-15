import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = prediction[0]
        return render_template('index.html', prediction_text=f'Predicted Crop: {output}')
    except Exception:
        return render_template('index.html', prediction_text='Error in prediction. Please check your input.')

def handler(request, *args, **kwargs):
    return app(request.environ, start_response=lambda *a, **k: None)
