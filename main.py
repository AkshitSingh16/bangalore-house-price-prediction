import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = int(request.form.get('bhk'))
        bath = int(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        print(location, bhk, bath, sqft)

        input_data = pd.DataFrame([[location, bhk, bath, sqft]],
                                   columns=['location', 'bhk', 'bath', 'total_sqft'])

        prediction = pipe.predict(input_data)[0] * 1e5
        return str(np.round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)