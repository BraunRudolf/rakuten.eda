import pandas as pd
import re
from tqdm import tqdm 

tqdm.pandas()

def merge_dataframes(df1, df2, how, on=None, index_true=False):
    if on is not None:
        if (df1[on].nunique() != df1[on].shape[0]) or (df2[on].nunique() != df2[on].shape[0]):
            raise ValueError("Column content is not unique.")
        return pd.merge(df1, df2, on=on, how=how)
    if index_true:
        return pd.merge(df1, df2, left_index=True, right_index=True, how=how)
    else:
        return pd.merge(df1, df2, on=on, how=how)

def fill_null_values(df, column_name, fill_value):
    df[column_name] = df[column_name].fillna(fill_value)
    return df

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

def class_distribution_abs(df, column_name):
    return df[column_name].value_counts()

def duplicated_column(df):
    duplicated_cols = {}
    for col in df.columns:
        duplicated_cols[col] = df[col].duplicated().sum()
    return duplicated_cols

def count_entries_with_same_text_different_prdtypecode(df, cols_short:list, cols_long:list):
    duplicated_products = df[cols_short].duplicated().sum()
    duplicated_products_prdtypecode = df[cols_long].duplicated().sum()
    return duplicated_products - duplicated_products_prdtypecode

def class_distribution_rel(df, column_name):
    return df[column_name].value_counts(normalize=True)

def count_punctuation(text):
    return len(re.findall(r'[^\w\s]', text))

def punctuation_ratio(df, column_name):

    return df.apply(lambda x: count_punctuation(x[column_name]) / len(x[column_name]), axis=1)

def string_lenght(df, column_name):
    return df[column_name].str.len()

def count_words(text, spacy_nlp):
    doc = spacy_nlp(text)
    # TODO: check if spacy is able to recognize 'w/o'
    return  sum(1 for token in doc if token.is_alpha)
    # TODO: both methods
    return len([token for token in doc if token.is_alpha])

def word_count(df, column_name, spacy_nlp):
    # NOTE: Spacy doesn't recognizes 'w/o' as a word
    # TODO: Check text for special words

    return df[column_name].progress_apply(lambda x: count_words(x, spacy_nlp))

# quick, rough count using the regex
def count_words_regex(text,pattern=r"\b[\w'-]+\b"):
    words = re.findall(pattern, text)
    return len(words)

def word_count_regex(df, column_name, pattern=r"\b[\w'-]+\b"):
    # NOTE: Spacy doesn't recognizes 'w/o' as a word
    # TODO: Check text for special words

    return df[column_name].progress_apply(lambda x: count_words_regex(x, pattern))


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


