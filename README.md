# Disaster Response Project

This web app showcases a machine learning model that is capable of classifying disaster-related help requests according to the message's content. The aim is to ensure that victims of large-scale disasters have their messages mapped to the right disaster response groups, either automatically or with minimal human intervention.

You can view the app on https://disaster-response-brunoarine.herokuapp.com

## How it works

A Python program extracts the text from translated, pre-labeled tweets, messages, and news snippets of real-life disasters from a dataset provided by [Appen](https://www.appen.com) in CSV format. The dataset is then cleaned and converted to SQL format, which is used to train 36 incremental learning algorithms—one for each of the 36 message categories. The models are put together and their hyperparameters optimized, and an app built on Flask and hosted on Heroku serves the content on the web. The algorithm takes the user's message as input, and returns a list of the most probable categories of help request to which the user's message belongs.

## How to run this project

First, type the following in your shell prompt:

```sh
git clone https://github.com/brunoarine/disaster-response.git
```
### Process the raw data into an SQL database

Open the `data` folder and type the following in your shell prompt:

```sh
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```
### Train the classifier

Open the `models` folder and type the following in your shell prompt:

```sh
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
### Run the web app

Open the `app` folder and type:

```sh
python run.py
```

You'll be able to access the server on your local machine at http://0.0.0.0:3001


## Prerequisites

- numpy 1.20.2
- plotly 4.14.3
- joblib 1.0.0
- Flask 1.1.2
- nltk 3.6
- SQLAlchemy 1.3.23
- pandas 1.2.3
- scikit_learn 0.24.2

Install the packages with:

```sh
pip install -r requirements.txt
```

## Acknowledgements

- [Udacity](https://www.udacity.com) for providing the tools and incentive to carry out this project.
- [Appen](https://www.appen.com) for providing a huge amount of labeled data, which was key to the success of this project.
