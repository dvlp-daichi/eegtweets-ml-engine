import json
import subprocess
from flask import jsonify, make_response
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

def predict_emotions(request):
    
    PROJECT_ID = "eegtweets"
    MODEL_NAME = "emotionanalysis"
    
    alpha = 0
    beta = 0
    theta = 0
    delta = 0
    gamma = 0
    
    request_json = request.get_json()
    
    if request_json:
    
        if 'alpha' in request_json:
            alpha = request_json['alpha']
    
        if 'beta' in request_json:
            beta = request_json['beta']
    
        if 'theta' in request_json:
            theta = request_json['theta']
    
        if 'delta' in request_json:
            delta = request_json['delta']
    
        if 'gamma' in request_json:
            gamma = request_json['gamma']
    
    
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build("ml", "v1", credentials=credentials)
    
    data = {
        "instances": [
            {"x": [alpha, beta, theta, delta, gamma], "key": 0}
        ]
    }
    
    print(data)
    
    req = ml.projects().predict(
        body=data,
        name="projects/{0}/models/{1}".format(PROJECT_ID, MODEL_NAME)
    )
    
    print(json.dumps(req.execute()))
    
    prediction = json.loads(json.dumps(req.execute()))['predictions'][0]['y']
    
    return make_response(jsonify( {
        "happiness": prediction[0],
        "surprise": prediction[1], 
        "disgust": prediction[2],
        "fear": prediction[3],
        "sadness": prediction[4],
        "anger": prediction[5]
    } ))