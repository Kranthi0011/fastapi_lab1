import pickle
import os
import numpy as np
from src.data import IrisData

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'iris_model.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def predict(data: IrisData) -> int:
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    prediction = model.predict(features)
    return int(prediction[0])
