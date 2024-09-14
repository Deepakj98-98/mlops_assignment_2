import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
#import autosklearn.classification
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, classification_reportp

# Load Dataset (Example: Titanic Dataset from Kaggle)
data = pd.read_csv("C:\\Users\\Deepak J Bhat\\Downloads\\train.csv")