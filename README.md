# **Near Earth Objects**
## **Project Overview**
This project involves analyzing asteroid data, specifically focusing on Near-Earth Objects (NEOs) and Potentially Hazardous Asteroids (PHAs). The dataset used in this project is sourced from a CSV file (asteroid.csv). The primary objectives are to:
1. Perform exploratory data analysis (EDA) on the dataset.
2. Extract subsets of the data for NEOs and PHAs.
3. Visualize the relationships between different parameters.
4. Implement and evaluate machine learning models to classify asteroids.

## **Table of Contents**
1. Importing Libraries and Data
2. Exploratory Data Analysis (EDA)
3. Data Preparation
4. Visualization
5. Model Selection and Analysis
6. Results
7. Conclusion
8. Future Work

## Importing Libraries and Data
To start with, necessary libraries are imported, and warnings are suppressed to keep the output clean.
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="ticks", color_codes=True)
```

The dataset is loaded from the asteroid.csv file:
```
ad = pd.read_csv('asteroid.csv')
print(ad.shape)
ad.head()
```


## Exploratory Data Analysis (EDA)
EDA involves understanding the structure of the data and extracting meaningful insights. Here, we focus on extracting subsets of data:

###Extracting NEOs and PHAs
Functions are defined to extract NEOs, PHAs, and the intersection of both:
```
def extract_neo(df):
    neo = df.loc[df.neo == 'Y', :]
    neo.to_csv('neo.csv')
    return neo

def extract_pha(df):
    pha = df.loc[df.pha == 'Y', :]
    pha.to_csv('pha.csv')
    return pha

def extract_neo_pha(df):
    neo_pha = df.loc[(df.neo == 'Y') & (df.pha == 'Y'), :]
    neo_pha.to_csv('neo_pha.csv')
    return neo_pha
```

### Scatter Plot Visualization
Scatter plots are created to visualize relationships between different parameters:
```
def plot_scatter(df, params):
    for i in params:
        g = sns.FacetGrid(df, col='class', hue='pha')
        g.map(sns.scatterplot, 'moid', i, alpha=.7)
        h = sns.FacetGrid(df, col='class', hue='pha')
        h.map(sns.scatterplot, 'H', i, alpha=.7)
        g.add_legend()
        h.add_legend()
        plt.show()
```


## Data Preparation
###Extracting and Cleaning Data
The data is filtered to include only rows with non-null values for pha and neo:
```new_ad = ad[ad['pha'].notna()]
new_ad = new_ad[new_ad['neo'].notna()]
```

###Identifying Important Parameters
Parameters of interest are identified and their summary statistics are generated:
```
param_imp = ['diameter', 'albedo', 'e', 'a', 'q', 'i', 'tp']
neo[param_imp].info()
neo[param_imp].describe()
```

###Handling Missing Values
Missing values are filled with the mean value of each column:
```
def fill_nan(df):
    for column in df.columns:
        if column in ['pha', 'neo']:
            df[column].fillna(value=0, inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)
    return df

neo_params_imp = fill_nan(neo_params_imp)
neo_params_all = fill_nan(neo_params_all)
```


## Visualization
Correlation heatmaps are generated to understand the relationships between parameters:
```
plt.figure(figsize=(20, 20))
sns.heatmap(data=round(neo_params_imp.corr(), 2), annot=True)
plt.show()

plt.figure(figsize=(20, 20))
sns.heatmap(data=round(neo_params_all.corr(), 2), annot=True)
plt.show()
```


## Model Selection and Analysis
Various machine learning models are trained and evaluated to classify asteroids based on their characteristics.
###Model Implementation
```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

def model_fit_score(models, df):
    X = df.drop('pha', axis=1)
    y = df['pha']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    model_scores = pd.DataFrame(model_scores, index=['Score']).transpose()
    model_scores = model_scores.sort_values('Score')
    return model_scores

models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}
```

###Model Evaluation
The models are evaluated and their scores are plotted:
```
model_scores_params_imp = model_fit_score(models, neo_params_imp)
model_scores_params_imp.sort_values('Score', ascending=False)

plt.figure(figsize=(20, 10))
sns.barplot(data=model_scores_params_imp.sort_values('Score').T)
plt.show()
```


## Results
The best-performing models based on their scores are:
1. GradientBoostingClassifier
2. SVC
3. RandomForestClassifier
These models demonstrate the highest accuracy in classifying PHAs.


## Conclusion
This project provides a comprehensive analysis of asteroid data, highlighting the importance of various parameters in identifying NEOs and PHAs. The machine learning models implemented offer valuable insights and tools for asteroid classification.

## Future Work
Future work could involve:
1. Exploring additional machine learning models and techniques.
2. Enhancing data preprocessing and feature engineering.
3. Incorporating more recent and diverse datasets for improved accuracy.
4. Deploying the best models in a real-time application for asteroid monitoring and detection.
