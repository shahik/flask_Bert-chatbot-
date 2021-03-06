# -*- coding: utf-8 -*-
"""Bert_Chatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uJj3HiIohJiOnYeBNUbNW3uAZsHNVCS0
"""

rm -r flask_Bert-chatbot-/

pip install --user --upgrade tensorflow

import tensorflow as tf
print(tf.__version__)

import os
os.mkdir('thesis')

!pip install cdqa  #Library for Bert

import pandas as pd
from ast import literal_eval

from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

# Download model
download_model(model='bert-squad_1.1', dir='./models')

# Download pdf files from BNP Paribas public news
def download_pdf():
    import os
    import wget
    directory = './data/pdf/'
    models_url = [
      'https://invest.bnpparibas.com/documents/1q19-pr-12648',
      'https://invest.bnpparibas.com/documents/4q18-pr-18000',
      'https://invest.bnpparibas.com/documents/4q17-pr'
    ]

    print('\nDownloading PDF files...')

    if not os.path.exists(directory):
        os.makedirs(directory)
    for url in models_url:
        wget.download(url=url, out=directory)

download_pdf()

import os
df = pdf_converter(directory_path='./data/pdf/')

#df = pdf_converter(directory_path='./thesis/')   
#df.to_csv('botdata.csv')

df = filter_paragraphs(df)

df.head()

cdqa_pipeline = QAPipeline(reader='./models/bert_qa_vCPU-sklearn.joblib', max_df=1.0)

# Fit Retriever to documents
cdqa_pipeline.fit_retriever(df)

query = ' What is time series?'
prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))                
print('paragraph: {}'.format(prediction[2]))

"""-------------- Model Deployment---------------------------"""

from sklearn.externals import joblib
joblib.dump(cdqa_pipeline, 'Chatbot.sav')

'''
run in gitbash 
!git clone https://github.com/shahik/flask_Bert-chatbot-.git
Go to cloned directory and add file to upload then 
!git add .
!git commit -m "commit"
!git push origin master
'''

!pip install flask-ngrok     
!pip install flask==0.12.2

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.externals import joblib

from flask_ngrok import run_with_ngrok

#Load model 
model = joblib.load('Chatbot.sav')

app = Flask(__name__)
run_with_ngrok(app)
# routes
@app.route('/')

def predict():

    result = model.predict(' What is time series ?')
    result
    # send back to browser
    output = {'results': result[0]}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run()