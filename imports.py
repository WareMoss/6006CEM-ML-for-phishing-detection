# imports.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocessing import X_train, y_train, X_test, y_test
from cleaningcsv import main as cleaningcsv_main
from preprocessing import main as preprocess_main
from MLRF import model_name as model_name_rf, y_test_rf, y_pred_rf, accuracy_rf, main as mlrf_main
from MLLR import model_name_LR, y_test_LR, y_pred_LR, accuracy_LR, main as mllr_main
#from MLKNN import model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN, main as MLKNN_main
from eval import main as eval_main
#from sklearn.neighbors import KNeighborsClassifier
