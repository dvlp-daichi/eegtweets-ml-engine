import json
import subprocess
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    """
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return f'Hello World!' 
    """

    PROJECT_ID = "eeg-pi"
    MODEL_NAME = "tictactoe"

    print("Calling ML model on Cloud ML Engine...")
    credentials = GoogleCredentials.get_application_default()
    ml = discovery.build("ml", "v1", credentials=credentials)
    data = {
        "instances": [
            {"x": [0, 0, 0, 0, 0, 0, 0, 0, 0], "key": 0}
        ]
    }

    req = ml.projects().predict(
        body=data,
        name="projects/{0}/models/{1}".format(PROJECT_ID, MODEL_NAME)
    )

    return json.dumps(req.execute(), indent=2)