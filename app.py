from flask import Flask, request, jsonify
from flask_cors import CORS 
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

rd = pd.read_csv("weather_forecast_data.csv")
rd['Rain'] = rd['Rain'].map({'rain': 1, 'no rain': 0})
x_train = rd.drop('Rain', axis=1)
y = rd['Rain']

x_mean = x_train.mean()
x_std = x_train.std()
x_scale = (x_train - x_mean) / x_std

w = np.zeros(x_scale.shape[1])
b = 0
alpha = 0.001
m = x_scale.shape[0]
iterations = 1000

for _ in range(iterations):
    z = np.dot(x_scale, w) + b
    y_train = 1 / (1 + np.exp(-z))
    cost = y_train - y
    gradient_W = (1/m) * np.dot(cost, x_scale)
    gradient_b = (1/m) * np.sum(cost)
    w = w - alpha * gradient_W
    b = b - alpha * gradient_b

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_scaled = (input_df - x_mean) / x_std
    z = np.dot(input_scaled, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    prediction = int(y_pred >= 0.5)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
