import numpy as np

# %%
 # import all the necessary libraries
import os
import re
import pandas as pd
import numpy as np

import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import discriminant_analysis
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import xgboost as xgb



# options
data_dir = '/Users/coupon/projects/propulsion/courses/Week-5-ML-Supervised/Day-5-Classifiers-2'
plt.style.use('seaborn-whitegrid')

plt.rc('pdf',fonttype=42)
sns.mpl.rc('figure', figsize = (10, 8))
sns.set_context('notebook', font_scale=1.8, rc={'lines.linewidth': 2.5})


# %%
print(np.pi)


var = 2.0

print(var)


var = 3.0


string = 'Plot this\n information'
print(string)


import matplotlib
matplotlib.use('Qt5Agg')
# This should be done before `import matplotlib.pyplot`
# 'Qt4Agg' for PyQt4 or PySide, 'Qt5Agg' for PyQt5
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 20, 500)
plt.plot(t, np.sin(t))
None
plt.show()

import numpy as np
import pandas as pd

df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                   'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                   'D': np.array([3] * 4, dtype='int32'),
                   'E': pd.Categorical(["test", "train", "test", "train"]),
                   'F': 'foo'})

df
