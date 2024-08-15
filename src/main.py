import functions as func
import spacy
import pandas as pd

# Load spacy
spacy_nlp = spacy.load("fr_core_news_sm")

# Load data
X_train = pd.read_csv("X_train_update.csv", index_col=0)
y_train = pd.read_csv("Y_train_CVw08PX.csv", index_col=0)

# Merge data
df = func.merge_dataframes(X_train, y_train, how='left', index_true=True)
column_names = ['designation', 'description']

print(df.head())
# Class distribution
class_distribution = func.class_distribution_abs(df, column_name="prdtypecode")
print(class_distribution)

class_distribution_rel = func.class_distribution_rel(df, column_name="prdtypecode")
print(class_distribution_rel)

# check duplicated columns
duplicated_columns = func.duplicated_column(df)
print(duplicated_columns)

# investiagte duplicateds_further
entreis_with_same_text_different_prdtypecode = func.count_entries_with_same_text_different_prdtypecode(df, column_names, column_names+['prdtypecode'])
print(f"WARNING: There are {entreis_with_same_text_different_prdtypecode} entries with the same text but different prdtypecode")


# Check null values
null_values = func.null_values(df)
print(null_values)

# Fill null values
df = func.fill_null_values(df, column_name="description", fill_value='')

# Combine string columns
df = func.combine_str_columns(df, column_names, new_column_name="text", sep=' ')
print(df.head())

# Count words
# word_count = func.word_count(df, column_name="text", spacy_nlp=spacy_nlp)
# print(word_count)

word_count_regex = func.word_count_regex(df, column_name="text")
print(word_count_regex)

# Count sentences
sentence_count = func.count_sentences(df, column_name="text", spacy_nlp=spacy_nlp)
print(sentence_count)

