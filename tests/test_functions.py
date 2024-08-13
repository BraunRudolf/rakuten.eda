import pytest
import pandas as pd

import src.functions as func

@pytest.fixture()
def dataframe_str_cols():
    return pd.DataFrame({"class": [1, 1, 2], "column1": ['a', 'b', 'c'], "column2": ['a', 'b', 'c']})

@pytest.fixture()
def dataframe_str_punct_cols():
    return pd.DataFrame({"class": [1, 1, 2], "column1": ['a!', ',b?', ':c:'], "column2": ['a...', 'b;;;', 'c????']})

@pytest.fixture()
def dataframe_int_cols() -> pd.DataFrame:
    return pd.DataFrame({"class": [1, 1, 2], "column1": [1, 2, 3], "column2": [1, 2, 3]})

@pytest.fixture()
def dataframe_none() -> pd.DataFrame:
    return pd.DataFrame({"class": [None, 1, 2], "column1": [1, None, 3], "column2": [1, 2, 3]})


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
    assert all(round(func.punctuation_ratio(dataframe_str_punct_cols, 'column2'),2) == pd.Series({0: 0.75, 1: 0.75, 2: 0.8}))
