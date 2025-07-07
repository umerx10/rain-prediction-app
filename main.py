import numpy as np
import pandas as pd

rd=pd.read_csv('weather_forecast_data.csv')

rd['Rain'] = rd['Rain'].map({'rain': 1, 'no rain': 0})
x_train=rd.drop('Rain',axis=1)
y=rd['Rain']


x_mean=np.mean(x_train)
x_std=np.std(x_train)
x_scale=(x_train-x_mean)/x_std

w=np.zeros(x_scale.shape[1])
m=(x_scale.shape[0])
alpha=0.001
b=0
iterations=1000

for _ in range(iterations):
    z=np.dot(x_scale,w)+b
    y_train=1/(1+np.exp(-z))
    cost=y_train-y
    gradient_W=(1/m) * (np.dot(cost,x_scale))
    gradient_b=(1/m) * np.sum(cost)
    w=w-alpha*gradient_W
    b=b-alpha*gradient_b

z=np.dot(x_scale,w)+b
y_pred=1/(1+np.exp(-z))
y_pred_labels =np.where(y_pred > 0.5, 1, 0)


accuracy=np.mean(y_pred_labels==y)
print(accuracy)


