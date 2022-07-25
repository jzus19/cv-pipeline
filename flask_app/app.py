# some utilities
import os
from cv2 import transform
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from utils import base64_to_pil
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..\src')
from model import get_model
from dataset import get_transformations
# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
#Model_json = ".json"
Model_weigths = ".pt"


# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    model = get_model()
    model.to("cuda:0")
    model.load_state_dict(torch.load("../weights/Loss1.0944_epoch13.pt"))
    model.eval()
    return model  
    


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    pseudo2real = dict(
                        n02086240= 'Shih-Tzu',
                        n02087394= 'Rhodesian ridgeback',
                        n02088364= 'Beagle',
                        n02089973= 'English foxhound',
                        n02093754= 'Australian terrier',
                        n02096294= 'Border terrier',
                        n02099601= 'Golden retriever',
                        n02105641= 'Old English sheepdog',
                        n02111889= 'Samoyed',
                        n02115641= 'Dingo'
                        )
    transforms = get_transformations()
    img = np.array(img)
    img = transforms["valid"](image=img)["image"].unsqueeze(0)
    prediction = model(img.to("cuda:0"))
    prediction = torch.argmax(prediction, dim=1)
    le = LabelEncoder()
    le.classes_ = np.load('../classes.npy')
    prediction = pseudo2real[le.inverse_transform(prediction.cpu().detach().numpy())[0]]
    return prediction


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        prediction = model_predict(img, model)
        print(prediction)
        # Serialize the result, you can add additional fields
        return jsonify(result=prediction)
    return None


if __name__ == '__main__':
    # app.run(port=5002)
    app.run()