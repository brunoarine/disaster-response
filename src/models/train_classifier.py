import sys
import pandas as pd

def load_data(database_filepath):
    """Loads the database in SQL format and returns machine-learning ready X,
    Y, and category_names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("messages", con=engine)

    # drop `id` and `original` columns since they have no effect on prediction
    df = df.drop(["id", "original"], axis=1)

    X = df[["message", "genre"]]
    Y = df.drop(["message", "genre"], axis=1)
    category_names = X.columns

    return X, Y, category_names


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
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()