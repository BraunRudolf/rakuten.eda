import pytest
import pandas as pd
import spacy

import src.functions as func

@pytest.fixture()
def dataframe_str_cols():
    return pd.DataFrame({"class": [1, 1, 2], "column1": ['a', 'b', 'c'], "column2": ['a', 'b', 'c']})

@pytest.fixture()
def dataframe_sentence_cols():
    return pd.DataFrame({"class": [1, 1, 2, 3],
                         "column1": ['This is a title', 'This is an other title', 'Last title', 'This is a title w/o description'],
                         "column2": [
                            'This is a description. Of a product.',
                            'This is an other description. Of a product. Which is good!',
                            'Last description. This is the product. Of a product.Awesome! Which you neeed?!',
                            ''
                         ]})

@pytest.fixture()
def dataframe_str_punct_cols():
    return pd.DataFrame({"class": [1, 1, 2], "column1": ['a!', ',b?', ':c:'], "column2": ['...', 'b;;;', 'c????']})

@pytest.fixture()
def dataframe_int_cols() -> pd.DataFrame:
    return pd.DataFrame({"class": [1, 1, 2], "column1": [1, 2, 3], "column2": [1, 2, 3]})

@pytest.fixture()
def dataframe_none() -> pd.DataFrame:
    return pd.DataFrame({"class": [None, 1, 2], "column1": [1, None, 3], "column2": [1, 2, 3]})

@pytest.fixture()
def nlp():
    nlp = spacy.load("en_core_web_sm")
    return nlp

@pytest.fixture()
def dataframe_htlml_cols():
    return pd.DataFrame({"class": [1, 1, 2, 3],
                         "column1": ['text', '<h1>text</h1>', '<h1>text</h1>', 'no html'],
                         "column2": ['<p>test</p> more text <p>test</p>',
                                     '<b>text</b>', '<b>text< not a tag', '<p> neither a tag is this > but this </p>']})

def test_combine_columns_str_col(dataframe_str_cols):
    print(dataframe_str_cols[['column1', 'column2']].dtypes)
    df = func.combine_str_columns(dataframe_str_cols, ['column1', 'column2'])
    assert df['combined'].tolist() == ['a a', 'b b', 'c c']

def test_combine_columns_int_col(dataframe_int_cols):
    with pytest.raises(ValueError, match="All specified columns must be of string type."):
        func.combine_str_columns(dataframe_int_cols, ['column1', 'column2'])

def test_combine_columns_none_value(dataframe_none):
    with pytest.raises(ValueError, match="Column has null values."):
        func.combine_str_columns(dataframe_none, ['column1', 'column2'])

def test_null_values_zero(dataframe_str_cols):
    assert all(func.null_values(dataframe_str_cols) == pd.Series({'class': 0, 'column1': 0, 'column2': 0}))

def test_null_values_one(dataframe_none):
    assert all(func.null_values(dataframe_none) == pd.Series({'class': 1, 'column1': 1, 'column2': 0}))

def test_class_distribution_ok(dataframe_str_cols):
    assert all(func.class_distribution(dataframe_str_cols, 'class') == pd.Series({1: 2, 2: 1}))

def test_class_distribution_none(dataframe_none):
    assert all(func.class_distribution(dataframe_none, 'class') == pd.Series({1: 1, 2: 1}))

def test_punctuation_ratio(dataframe_str_punct_cols):
    assert all(round(func.punctuation_ratio(dataframe_str_punct_cols, 'column1'),2) == pd.Series({0: 0.5, 1: 0.67, 2: 0.67}))
    assert all(round(func.punctuation_ratio(dataframe_str_punct_cols, 'column2'),2) == pd.Series({0: 1, 1: 0.75, 2: 0.8}))

def test_string_lenght(dataframe_str_cols, dataframe_str_punct_cols, dataframe_none):
    assert all(func.string_lenght(dataframe_str_cols, 'column1') == pd.Series({0: 1, 1: 1, 2: 1}))
    assert all(func.string_lenght(dataframe_str_cols, 'column2') == pd.Series({0: 1, 1: 1, 2: 1}))

    assert all(func.string_lenght(dataframe_str_punct_cols, 'column1') == pd.Series({0: 2, 1: 3, 2: 3}))
    assert all(func.string_lenght(dataframe_str_punct_cols, 'column2') == pd.Series({0: 3, 1: 4, 2: 5}))
    
def test_string_lenght_none(dataframe_none):
    with pytest.raises(AttributeError, match="Can only use .str accessor with string values!"):
        func.string_lenght(dataframe_none, 'column1')

def test_word_count(dataframe_sentence_cols, nlp):
    # NOTE: Spacy doesn't recognizes 'w/o' as a word, so 'w/o' isn't counted (hence 'w/o' is 0/ 3:5 should be 3:6)
    assert all(func.word_count(dataframe_sentence_cols, 'column1', nlp) == pd.Series({0: 4, 1: 5, 2: 2, 3: 5}, name='column1'))
    assert all(func.word_count(dataframe_sentence_cols, 'column2', nlp) == pd.Series({0: 7, 1: 11, 2: 13, 3: 0}, name='column2'))

def test_sentence_count(dataframe_sentence_cols, nlp):
    assert all(func.count_sentences(dataframe_sentence_cols, 'column1', nlp) == pd.Series({0: 1, 1: 1, 2: 1, 3: 1}, name='column1'))
    assert all(func.count_sentences(dataframe_sentence_cols, 'column2', nlp) == pd.Series({0: 2, 1: 3, 2: 5, 3: 0}, name='column2'))
    

def test_html_tag_count(dataframe_htlml_cols):
    assert all(func.html_tag_count(dataframe_htlml_cols, 'column1') == pd.Series({0: 0, 1: 2, 2: 2, 3: 0}, name='column1'))
    assert all(func.html_tag_count(dataframe_htlml_cols, 'column2') == pd.Series({0: 4, 1: 2, 2: 1, 3: 2}, name='column2'))

def test_group_by_class(dataframe_htlml_cols):
    dataframe_htlml_cols['num_html_tags'] = func.html_tag_count(dataframe_htlml_cols, 'column1')
    print(dataframe_htlml_cols.head())
    assert all(dataframe_htlml_cols.groupby('class')['num_html_tags'].sum() == pd.Series({1: 2, 2: 2, 3: 0}))
