import numpy as np
from flask import Flask, request, jsonify, render_template  
import pickle

flask_app= Flask(__name__)
model=pickle.load(open("crop_model.pkl", "rb"))

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = prediction[0]
        return render_template('index.html', prediction_text=f'Predicted Crop: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text='Error in prediction. Please check your input.')
    
if __name__ == "__main__":
    flask_app.run(debug=True)   

