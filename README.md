# prediction-probabilityofdefault
==============================

Prediction the probability of default

Table of contents
* [Objective](#objective)
* [Development](#development)
* [Project Structure](#project-structure)
* [API](#api)
* [Installation](#installation)
* [Environment](#environment)
* [Validation](#validation)

## Objective
The aim of the case study is the prediction of the probability of default for the data points in the given file. 

## Development
In the model development phase, first, the data is imported and start the analysis process. The main steps are listed below:

-Target analysis, missing imputation, numerical and categorical feature analysis, correlation analysis, OneHotEncoder, and SMOTE steps are performed. Accordingly, the feature list is decided, and hyperparameter tuning is run. 

## Project Structure
~~~
app
├── data
│   └── input
│       └── dataset.csv
├── models
│   ├── final_model.pkl
│   └── ohe.pkl
├── notebooks
│   └── modules
│       ├──dataanalysis.py
│       ├──missingimputation.py
│       ├──ohe_functions.py
│       └──run_model_metrics.py
│   ├──api.py
│   ├──CaseStudy.ipynb
│   ├──csvprocess.py
│   ├──dataprocess.py
│   ├──model.py
│   └──variables.py
│── task 
├── app.py
├── dockerfile
├── README.md
└── requirements.txt

~~~

## API
In the `app.py` script, a Flask API is developed to predict the default for the rows which don't have a target.
The app id dockerized as seen in the Dockerfile.

## Installation
Build the docker image:

~~~~python
docker build -t project .
~~~~

Run the docker image:
~~~~python
docker run -d -p 5000:5000 project
~~~~

Copy to below to the browser:
~~~~python
http://localhost:5000/predictions
~~~~
Or you can also download this repo, then import directly into Postman.

After running, the `defaultPredictions.csv` will be download in the requested format.

## Environment
Create an environment with requirements.txt
~~~~
conda env create --file requirements.txt 
~~~~

## Validation
PSI values of the features can be calculated

