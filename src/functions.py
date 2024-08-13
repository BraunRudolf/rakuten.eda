import pandas as pd
import re

def combine_str_columns(df, column_names, new_column_name="combined", sep=' '):
    """
    Combines multiple string columns into a single column
    """
    try:
        if df[column_names].isnull().values.any():
            raise ValueError("A Column has null values.")

        if any(df[column_names].dtypes != 'object'):
            raise ValueError("All specified columns must be of string type.")

        df[new_column_name] = df[column_names].apply(lambda x: sep.join(x), axis=1)
    
    except Exception as e:
        print(e)
        raise e
    return df


def null_values(df):
    return df.isnull().sum()

def class_distribution(df, column_name):
    return df[column_name].value_counts()

def count_punctuation(text):
    return len(re.findall(r'[^\w\s]', text))

def punctuation_ratio(df, column_name):

    return df.apply(lambda x: count_punctuation(x[column_name]) / len(x[column_name]), axis=1)

def string_lenght(df, column_name):
    return df[column_name].str.len()

def word_count(df, column_name):
    return df[column_name].str.split().str.len()

def sentence_count(df, column_name):
    return df[column_name].str.count(r'[.!?]')

def html_tag_count(df, column_name):
    return df[column_name].str.count(r'<[^>]+>')

def group_by_class(df, class_col, func):
    return df.groupby(class_col).apply(func)


