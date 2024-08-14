import pandas as pd
import re

# TODO: return new columns instead of results
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

def count_words(text, spacy_nlp):
    doc = spacy_nlp(text)
    return len([token for token in doc if token.is_alpha])

def word_count(df, column_name, spacy_nlp):
    # NOTE: Spacy doesn't recognizes 'w/o' as a word
    # TODO: Check text for special words
    return df[column_name].apply(lambda x: count_words(x, spacy_nlp))

def count_sentence_in_text(spacy_nlp, text):
    if text is None:
        return 0
    doc = spacy_nlp(text)
    return len(list(doc.sents))

def count_sentences(df, column_name, spacy_nlp):
    return df[column_name].apply(lambda x: count_sentence_in_text(spacy_nlp, x))

def html_tag_count(df, column_name):
    # TODO: compare to bs4 implementation
    return df[column_name].str.count(r'<[^>]+>')

def group_by_class(df, class_col, group_cols, func:list):
    return df.groupby(class_col).apply(lambda x: x[group_cols].agg(func))


