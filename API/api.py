from __future__ import division, print_function
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
import joblib
import json

from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = '../models/lr_pca_model.joblib'

@app.route('/api/v0/verify', methods=['GET', 'POST'])
def predict_test():
	vec = request.get_json()
	print('Received params:', vec)
	loaded_model = joblib.load(MODEL_PATH)
	test_vec = np.array(vec['v'])
	test_vec = test_vec.reshape(-1,1)
	res = str(loaded_model.predict(test_vec)[0])

	result = [
    	{
    		'id': 0,
     		'prediction': res
     	}
    ]

	return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)