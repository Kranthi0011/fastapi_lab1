# FastAPI Lab 1 - Iris Classifier API

## Overview
This project is part of the MLOps course at Northeastern University. It demonstrates how to expose a Machine Learning model as a REST API using FastAPI and Uvicorn. A Decision Tree Classifier is trained on the famous Iris dataset and served as a production-ready API with proper request validation, error handling, and interactive documentation.

## Objectives
- Train a Decision Tree Classifier on the Iris dataset
- Save the trained model as a pickle file
- Build a REST API using FastAPI to serve predictions
- Validate incoming requests using Pydantic models
- Test the API using the built-in Swagger UI documentation

## Tech Stack
- Python 3.8+
- FastAPI - Modern, high-performance web framework for building APIs
- Uvicorn - Lightning-fast ASGI web server
- Scikit-learn - Machine learning library for training the model
- Pydantic - Data validation and settings management
- Pickle - Model serialization and deserialization
- NumPy - Numerical computing

## Project Structure
fastapi_lab1/
├── model/
│   └── iris_model.pkl        # Trained Decision Tree model
├── src/
│   ├── __init__.py           # Makes src a Python package
│   ├── data.py               # Pydantic models for request and response
│   ├── main.py               # FastAPI application with route handlers
│   ├── predict.py            # Model loading and prediction logic
│   └── train.py              # Model training and saving logic
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

## Dataset
The Iris dataset contains 150 samples of iris flowers with the following features:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

Target Classes:
- 0 = Iris Setosa
- 1 = Iris Versicolor
- 2 = Iris Virginica

## Setup Instructions

### Step 1: Clone the repository
git clone https://github.com/Kranthi0011/fastapi_lab1.git
cd fastapi_lab1

### Step 2: Create and activate virtual environment
python -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate

### Step 3: Install dependencies
pip install -r requirements.txt

### Step 4: Train the model
python -m src.train

### Step 5: Start the API server
uvicorn src.main:app --reload

### Step 6: Test the API
Open your browser and go to: http://127.0.0.1:8000/docs

## API Endpoints

### GET /
Returns a welcome message.
Response: { "message": "Welcome to the Iris Classifier API! Go to /docs to test the endpoints." }

### POST /predict
Accepts iris flower measurements and returns the predicted species class.

Example Request:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Example Response:
{
  "response": 0
}

## Data Models

### IrisData (Request Model)
- sepal_length: float - Length of the sepal in cm
- sepal_width: float - Width of the sepal in cm
- petal_length: float - Length of the petal in cm
- petal_width: float - Width of the petal in cm

### IrisResponse (Response Model)
- response: int - Predicted class (0=Setosa, 1=Versicolor, 2=Virginica)

## Model Details
- Algorithm: Decision Tree Classifier
- Dataset: Iris (150 samples, 4 features, 3 classes)
- Train/Test Split: 80% training, 20% testing
- Test Accuracy: 100%
- Model Storage: Serialized using Python pickle

## Error Handling
- 422 Unprocessable Entity: Returned when request data is invalid or missing fields
- 500 Internal Server Error: Returned when prediction fails unexpectedly

## Author
Kranthi Kiran
Northeastern University
Course: MLOps (IE 6400)
