import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """Loads and transforms raw data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # concatenate the dataframes on the id
    df = pd.merge(messages, categories, how="inner", on="id")

    # expand the semi-collon categories into new columns
    categories = df.categories.str.split(";", expand=True)

    # extract a list of new column names for categories
    row = categories.loc[0]
    category_colnames = [x.split("-")[0] for x in row.values]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    # e.g. related-0 becomes 0, related-1 becomes 1
    for column in categories:
        categories[column] = categories[column].str.replace(f"{column}-", "")
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, how="outer", left_index=True, right_index=True)

    return df


def clean_data(df):
    """Cleans data by removing duplicates and possible glitches from the original data set.
    """

    # drop duplicates
    df.drop_duplicates(subset=None, keep="first", inplace=True)

    # # drop rows where related = 2 (probably a glitch)
    df = df[df["related"] != 2]

    # drop "child_alone" category, there are no samples that fit this one
    df.drop("child_alone", axis=1, inplace=True)

    return df


def save_data(df, database_filepath):
    """Dumps Pandas dataframe into an SQL database.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')

    df.to_sql('messages', engine, index=False, if_exists="replace")

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              '../../data/raw/messages.csv ../../data/raw/categories.csv '\
              '../../data/processed/messages.db')


if __name__ == '__main__':
    main()