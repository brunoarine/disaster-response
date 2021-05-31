import re
import sys

import joblib
import numpy as np
import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sqlalchemy import create_engine
import multiprocessing


# estimate number of cores to optimize model training (i.e. use total available cpus - 1)
CPU_COUNT = multiprocessing.cpu_count()

# download nltk data
download('punkt')
download('stopwords')

# load nltk's stopwords into a global variable instead of
# calling the function repeatedly
STOPWORDS = stopwords.words("english")


def load_data(database_filepath):
    """Loads the database in SQL format and returns machine-learning ready X,
    Y, and category_names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("messages", con=engine)

    # drop `id` and `original` columns since they have no effect on prediction
    df = df.drop(["id", "original"], axis=1)

    X = df["message"]
    Y = df.drop(["message", "genre"], axis=1)

    return X, Y


def tokenize(text):
    """Tokenizes texts by first stripping punctuations and normalizing cases.
    Words are also preprocessed with Porter Stemmer before tokenization takes place.
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = regexp_tokenize(text, pattern="\w+")

    tokens = [word for word in tokens if word not in STOPWORDS]
    stemmer = PorterStemmer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = stemmer.stem(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds the machine learning model
    """

    pipeline = make_pipeline(
        TfidfVectorizer(
            tokenizer=tokenize,
            max_df=0.5,
            ngram_range=(1, 2),
            use_idf=True),
        VarianceThreshold(),
        MaxAbsScaler(),
        MultiOutputClassifier(
            PassiveAggressiveClassifier(
                class_weight="balanced",
                early_stopping=True,
                random_state=33634
            )
        )
    )

    parameters = {
        'tfidfvectorizer__max_df': [0.4, 0.5, 0.6],
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__norm': ['l1', 'l2'],
        'tfidfvectorizer__use_idf': [True, False],
        'tfidfvectorizer__sublinear_tf': [True, False],
        'multioutputclassifier__estimator__C': [0.01, 0.025, 0.05, 0.075, 0.1]
    }

    grid = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3, verbose=1, n_jobs=CPU_COUNT-1)

    return grid


def evaluate_model(model, X_test, Y_test):
    """Return Precision-Recall AUC and ROC AUC
    """

    Y_pred = model.predict(X_test)
    scores = []
    for i, column in enumerate(Y_test.columns):
        scores.append(f1_score(Y_test[column].values, Y_pred[:, i]))
        print(str(column))
        print("-"*40)
        print(classification_report(Y_test[column].values, Y_pred[:, i]))

    print("Mean F1 score:", np.mean(scores))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath, compress=3)
    print('Trained model saved!')


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=33634)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.joblib')


if __name__ == '__main__':
    main()
