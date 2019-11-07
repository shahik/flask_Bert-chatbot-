import pandas as pd
from flask import Flask, jsonify, request
from sklearn.externals import joblib

from flask_ngrok import run_with_ngrok

#Load model 
model = joblib.load('Chatbot.sav')

app = Flask(__name__)
run_with_ngrok(app)
# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    #data = {'Pclass': 'What is time series ?'}

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run()