import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv", encoding='utf-8')

print(data.head().to_string())


# let's look that dataset has null value or not..!
# data.isnull.sum()

# let’s have a look at all the languages present in this dataset
# data["languages"].values_count()