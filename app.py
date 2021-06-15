import io
import json
import os
import random
import requests


from flask.templating import render_template_string

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import numpy as np 
#from flask import Flask, jsonify, request, 
#from flask_bootstrap import Bootstrap
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")


# nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
app = Flask(__name__)

#bootstrap = Bootstrap(app)
#model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
#model = torch.load('model.pth')
class MyClassifier(nn.Module):
    """
    Fully conneceted classifier we will train to predict skin moles from images
    Inputs:
        input_size: Depending on the model
        output_size: Depending on the problem (2 classes in this case)
        hidden_layers: The user can choose the number of ReLU hidden layers.
        dropout_p: Probability of dropout.
    """
    def __init__(self, input_size, output_size, hidden_layers,dropout_p):
        super().__init__()
        # The first layer 
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # We pair the rest of the layers and define them
        paired_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        for p1,p2 in paired_layers:
            self.hidden_layers.append(nn.Linear(p1,p2))
        # Define the output layer
        self.output_layer = nn.Linear(hidden_layers[-1],output_size)

        # Define that we will be using dropout - We will also check that it is between 0 and 1
        try:
            self.dropout = nn.Dropout(p=dropout_p)
        except ValueError:
            print("The dropout probability has to be between 0 and 1 amd got ",dropout_p)
            print("Please introduce a valid p")
            sys.exit("Program terminating.")


    # Then we define the forward method
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        x = F.log_softmax(x,dim=1)
        return x
    
def build_model():
    
    """
    Inputs:
        possible_models: A dictionary with the models that this app uses.
        
    Outputs:
        model: Pretrained model with the classifier defined by the user
        device: It selects whether the model is trained in CPU/GPU
    
    """
    # Loading pretrained vgg16 model
    model = models.vgg16(pretrained = True)

    #model = models.alexnet(pretrained = True) 
    #        
    # From here we can indicate not to compute the gradient
    for param in model.parameters():
        param.requieres_grad = False

    # Now we attach our classifier
    input_size = 25088
    output_size = 2
    classifier = MyClassifier(input_size, output_size, [1024,512], 0.2)
    model.classifier = classifier
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #model = model.to(device)
    
    return model, device

#skinmodel, device = build_model()
#skinmodel.load_state_dict(torch.load('model_weightskin.pth'))
model, device = build_model()
model.load_state_dict(torch.load('model_weights2.pth'))
model.eval()                                             


img_class_map = {0: 'Melanoma', 1: 'Not Melanoma'}





# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(256),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg


# Get a prediction
def get_prediction(input_tensor):
    # outputs1 = skinmodel.forward(input_tensor)                 # Get likelihoods for if picture of skin 
    # _, y_hat1 = outputs1.max(1)                             # Extract the most likely class
    prediction1 = 1# y_hat1.item() 
    #print('Prediction 1: ', prediction1)
    if prediction1 == 1:
        outputs2 = model.forward(input_tensor) 
        pred_probabilities = F.softmax(outputs2).data.squeeze()                # Get likelihoods for all classes
        _, y_hat2 = outputs2.max(1)                             # Extract the most likely class
        prediction2 = y_hat2.item()     
        #print('Prediction 2: ', prediction2)    
        #print('Prediction Probs: ', pred_probabilities.flatten().tolist())                    # Extract the int value from the PyTorch tensor
        return prediction2, pred_probabilities.flatten().tolist()
    return 99, _

# Make the prediction human-readable
def render_prediction(prediction_idx):
    #stridx = str(prediction_idx)
    class_name = 'This does not seem to be the picture of a skin lesion. Try again'
    if img_class_map is not None:
        if prediction_idx in img_class_map is not None:
            class_name = img_class_map[prediction_idx]

    return prediction_idx, class_name


# @app.route('/keynote', methods=['GET', 'POST'])
# def keynote():
#     return render_template('keynote.html')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('me.html')


@app.route('/', methods=['GET', 'POST'])
def demo():
    return render_template('demo.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx, probs = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            
            if prediction_idx==99:
                
                return render_template('99.html', class_name=class_name)
            else:
                class_names = list(img_class_map.values())
                #print(class_names)
                probs = [round(elem, 2) for elem in probs]
                if class_name == class_names[0]:
                    prob = probs[0]
                else:
                    prob = probs[1]
                return render_template('result.html', class_name=class_name, prob=prob,  file=file)
            

@app.route('/predicturl', methods=['GET','POST'])
def predicturl():
    input_transforms = [transforms.Resize(256),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    if request.method == 'POST':
        url = request.form.get('img_url')
        #print(type(url))
        print("URL: ", url)
        #print(url.keys())
        if url is not None:
            #response = requests.get(url)
            #img = Image.open(io.BytesIO(response.content))
            img = Image.open(requests.get(url, stream=True).raw)
            #type(img)
            img = img.resize((256,256), Image.ANTIALIAS)
            img_arr = np.array(img)
            #img_tensor = torch.from_numpy(img_arr)

            input_tensor = my_transforms(img)                           # Transform PIL image to appropriately-shaped PyTorch tensor
            input_tensor.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
            #return timg
            #input_tensor = transform_image(file)
            prediction_idx, probs = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            if prediction_idx==99:
                
                return render_template('99.html', class_name=class_name)
            else:
                class_names = list(img_class_map.values())
                #print(class_names)
                probs = [round(elem, 2) for elem in probs]
                if class_name == class_names[0]:
                    prob = probs[0]
                else:
                    prob = probs[1]
                #print(round(prob, 2))
                return render_template('result.html', class_name=class_name, prob=prob)

@app.route("/question")
def home():
    return render_template('question.html')



if __name__ == '__main__':
    app.debug = True
    app.run()