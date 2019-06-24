import os
os.sys.path.append("..")
from flask import Flask, render_template, request, jsonify
from predict import predict
import json

from model import vgg11_bn
import torch

app = Flask(__name__)

weight_path = '../vgg_model_PHF.pkl'
model = vgg11_bn()

try:
  checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
except FileNotFoundError:
  print("no such file")
else:
  model.load_state_dict(checkpoint["model"])
  print("Model loaded!")


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def get_ans():
  file = request.files["upfile"]
  filepath = './answer/'+file.filename 
  file.save(filepath)
  ans = predict(filepath, model)
  return ans

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)