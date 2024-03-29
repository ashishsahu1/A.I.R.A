![Web 1920 – 4](https://user-images.githubusercontent.com/52550558/114520241-ddfaeb80-9c5e-11eb-937b-04f544e7bc89.png)

## 📌About
AIRA is my personal AI based Resume Bot which can talk about me (Ashish) and provide important informations. AIRA is also enchanced with conversation intents to make the conversation feel more natural.

## 📌Requirements
- Flask : 1.1.2
- Flask-SQLAlchemy : 2.5.1
- Jinja2 : 2.11.3
- joblib : 1.0.1
- Keras-Preprocessing : 1.1.2
- nltk : 3.5
- numpy : 1.19.5
- sklearn-pandas : 2.0.4
- SQLAlchemy : 1.4.3
- tensorboard : 2.4.1
- tensorflow : 2.4.1
- tflearn : 0.5.0

## 📌Folder Manual
```
├───Artifacts
|   ├───checkpoint
|   ├───data.pickle
|   ├───model.tflearn.data-00000-of-00001
|   ├───model.tflearn.index
|   └───model.tflearn.meta
├───Intents
|   ├───intents.json
|   └───testIntents.json
├───Model
|   └───main.py
└───Server
    ├───static
    │   └───Images
    |   └───static.css
    |   └───static.js
    ├───templates
    |   └───index.html
    └───app.py
```


```js
   const myvar = "Manual"
    Artifacts : [	- All built pickles and other files,
                 	- tflearn, pickle exports
  	],
    Intents : [	- Conversation samples for the model,
                "Intents.json" : [ - Current working intent],
                "testIntents.json" : [ - under development intents],
  	],
    Model : [	- Creation of model,
                 	- tflearn, NLTK and other packages used
  	],
    Server : [	- Main server file,
                 	- Flask server setup
  	],
```



## 📌Screen Snaps
![snap](https://user-images.githubusercontent.com/52550558/143071480-bc89174f-a6e7-467b-9997-f30c913b7cb4.png)


## 📌Working Snap
![Chat with 💙AIRA - Google Chrome 2021-11-23 22-45-40 (2)](https://user-images.githubusercontent.com/52550558/143168115-de49683a-5561-4cab-9db1-aff9be87929e.gif)

## 📌Setup

- Clone the repo
    <br>

    ```
    git clone https://github.com/ashishsahu1/A.I.R.A.git
    ```

- Install minimum requirements
    <br>
    
    ```
    pip install -r requirements.txt
    ```
- Get inside Server
    <br>
    
    ```
    cd .\Server\
    ```
- Run Server
    <br>
    
    ```
    python app.py
    ```
- Open Localhost : http://localhost:5000/
    
<br>
<br>

<center><p>Made with 💜 and 🧠 by, <a href="https://www.linkedin.com/in/ashishsahu2/">Ashish 😊</a></p></center>


