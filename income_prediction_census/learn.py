import pandas as pd
import sys

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)


def convertLabel(x):
    """
    x could be ">50K", "<=50K" or "<=50K." (WTF?)
    """
    return -1 if x.replace(".","").strip() == '<=50K' else 1


columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation", "relationship",
    "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "label"
]

na_values = {
    'workclass': ' ?',
    'occupation': ' ?',
    'native_country': ' ?'
}

train = pd.read_csv("data/adult.data.csv", header=None, names=columns)
test = pd.read_csv("data/adult.test.csv", header=None, names=columns)

all_data = pd.concat([train, test], ignore_index=True)
all_data = all_data.dropna()

"""
train = train.dropna()
test = test.dropna()
"""

# Move the label from the dataframe to its own variable
all_data['label'] = all_data['label'].apply(convertLabel)
target = all_data.pop('label').values

"""
train['label'] = train['label'].apply(convertLabel)
test['label'] = test['label'].apply(convertLabel)
Y_train = train.pop('label').values
Y_test = test.pop('label').values
"""

kinds = np.array([dt.kind for dt in all_data.dtypes])
is_num = kinds != 'O'
all_columns = all_data.columns.values
num_cols = all_columns[is_num]
cat_cols = all_columns[~is_num]

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Convert numeric columns to float type to avoid warning:
# "DataConversionWarning: Data with input dtype int64 were all converted to float64 by StandardScaler."
"""
train = train.astype({c: np.float64 for c in num_cols})
test = test.astype({c: np.float64 for c in num_cols})
"""
all_data = all_data.astype({c: np.float64 for c in num_cols})


############ One hot encoding of categorical columns
cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,
                    handle_unknown='ignore'))
cat_steps = [cat_ohe_step]
cat_pipe = Pipeline(cat_steps)

############ Scaling numeric columns


#num_si_step = ('si', SimpleImputer(strategy='median'))

# Standardization of a dataset is a common requirement for many machine learning estimators:
# they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
num_ss_step = ('ss', StandardScaler())
num_steps = [num_ss_step]
num_pipe = Pipeline(num_steps)

transformers = [
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
]
ct = ColumnTransformer(transformers=transformers)


"""
X_train = ct.fit_transform(train)
X_test = ct.transform(test)
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2')
#
#clf.fit(X_train, Y_train)
#print(clf.score(X_test, Y_test))

ml_pipe = Pipeline([('transform', ct), ('clf', clf)])

#ml_pipe.fit(train, Y_train)
#score = ml_pipe.score(test, Y_test)

scores = cross_val_score(ml_pipe, all_data, target, cv=5)
print(scores)
