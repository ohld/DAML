###I use this snippets for the first look at the data to understand what is going on before dive into xgboost.

My preamble for all notebooks

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
```

Plot the amount of unique values in each feature to find categorical ones

```python
plt.figure(figsize=(10, 5))
sns.countplot(sorted([data[col].nunique() for col in data])[:250])
plt.title('Unique values in columns')
```

Print categorical features with unique values

```python
for name in list(data.columns.values):
    if len(data[name].unique()) <= 10:
        print name, "\n", data[name].unique(), "\n"
```

Quick look at the numerical features' importances.

```python
data_num = data[numerical].fillna(0)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(data_num, target)
importances = pd.DataFrame({'column':data_num.columns, 'importance':rf.feature_importances_}).sort_values(by='importance', ascending=0)
plt.figure(figsize=(15, 6))
sns.barplot(x='column', y='importance', data=importances.iloc[:20])
_ = plt.xticks(rotation=90)
```
