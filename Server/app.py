from flask import Flask, render_template, request
import json
import pickle
import numpy as np
import tflearn
import tensorflow
import random
import nltk
from nltk.stem import LancasterStemmer
stemer=LancasterStemmer()

#flask app
app = Flask(__name__)

#opening necessory files
with open("../Intents/intents.json") as file:
    data = json.load(file)

with open("../Artifacts/data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)

#Deep learning model
tensorflow.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation = "softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load("../Artifacts/model.tflearn")

#function for chat
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat(inputMsg):
    print("\n ------- Start Talking with AIRA ! ---------")
    while True:
        inp = inputMsg
        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(result)
        tag = labels[results_index]

        if result[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            
            return random.choice(responses)
        else:
            return "Sorry, I didn't get that, can you be more specific..."


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')   
    return str(chat(userText)) 

if __name__ == "__main__":    
    app.run(debug = True)