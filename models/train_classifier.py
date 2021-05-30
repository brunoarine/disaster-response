import sys
import pandas as pd
import joblib
import re
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import f1_score
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from nltk import download
from nltk.tokenize import regexp_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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

def multi_f_classif(X,y):
    """Extends the f_classif function to multioutput problems.
    """
    scores = []
    y = np.array(y)
    m = y.shape[1]
    for j in range(m):
        scores.append(f_classif(X, y[:,j])[0])
    return np.mean(scores, axis=0)


def build_model():
    """Builds the machine learning model
    """
  
    pipeline = make_pipeline(
                TfidfVectorizer(
                    tokenizer=tokenize,
                    max_df=0.5, 
                    max_features=5000,
                    use_idf=False),
                VarianceThreshold(),
                SelectKBest(multi_f_classif, k=500),
                MaxAbsScaler(),
                MultiOutputClassifier(
                    PassiveAggressiveClassifier(
                        C=0.02,
                        class_weight="balanced",
                        early_stopping=True,
                        random_state=33634
                )
        )
    )
    
    return pipeline


def evaluate_model(model, X_test, Y_test):
    """Return Precision-Recall AUC and ROC AUC
    """

    Y_pred = model.predict(X_test)
    scores = []
    for i, column in enumerate(Y_test.columns):
        scores.append(f1_score(Y_test[column].values, Y_pred[:,i]))
    
    print("Mean F1 score:", np.mean(scores))
    print(pd.DataFrame(np.array([Y_test.columns, scores]).T, columns=["Category", "F1 score"]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath, compress=3) 
    print('Trained model saved!')


def main():
    if len(sys.argv) == 3:
        # download nltk data
        download('punkt')
        download('stopwords')

        # load nltk's stopwords into a global variable instead of
        # calling the function repeatedly
        global STOPWORDS
        STOPWORDS = stopwords.words("english")

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.joblib')


if __name__ == '__main__':
    main()