# disaster-response

This web app showcases a machine learning model that is capable of classifying disaster-related help requests according to the message content. The aim is to ensure that victims of large-scale disasters have their messages mapped to the right disaster response groups, automatically or with minimal human intervention.

## How it works

A Python program extracts pre-labeled tweets, messages, and news snippets of real-life disasters from a dataset provided by Appen in CSV format. The dataset is then cleaned and converted to SQL format, which is used to train 36 incremental learning algorithms --- one for each of the 36 message categories. The models are put together and their hyperparameters optimized, and an app built on Flask and hosted on Heroku serves the content on the web. The algorithm takes the user's message as input, and returns a list of the most probable categories of help request to which the user's message belongs.

## How to run this project

## Requirements

- numpy 1.20.2
- plotly 4.14.3
- joblib 1.0.0
- Flask 1.1.2
- nltk 3.6
- SQLAlchemy 1.3.23
- pandas 1.2.3
- scikit_learn 0.24.2

## Acknowledgements

- Udacity for providing the tools and incentive to carry out this project.
- Appen for providing a huge amount of labeled data, which was key to the success of this project.
