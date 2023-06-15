import uvicorn
import pickle
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI

with open('model.pickle','rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hey look it\'s working'}

@app.get('/predict')
def predict(feature_1: int = 9, feature_2: int = 7, feature_3: int = 5):
    return {'output': model.predict([[feature_1, feature_2, feature_3]]).tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
