import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline,make_pipeline	 #for pipeline
from sklearn.compose import ColumnTransformer		 #for transformers
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#load the data
heart_df=pd.read_csv('data/heart.csv')
heart_df.info()
heart_df.drop(['age','chol'], axis='columns', inplace=True)
#split data in training and test set
X_train,X_test,y_train,y_test = train_test_split(heart_df.drop(columns=['target']),
                                               heart_df['target'],test_size=0.2,random_state=20)
heart_df.info()
#standard scaler
std_scaler= ColumnTransformer([('standard_scaler',StandardScaler(),[0,1,2,3,4,5,6,7,8,9,10])
],remainder='passthrough')
#normalize the data using yeo-johnson power transformer
power_transformer= ColumnTransformer([
    ('yeo_john_trans',PowerTransformer(standardize=True,copy=False),[0,1,2,3,4,5,6,7,8,9,10])
],remainder='passthrough')
#polynomial factor
poly = PolynomialFeatures(degree=3,include_bias=True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
#logistic regression
log_reg_model = LogisticRegression()
#Feature Selection:
# Filter based method: We'll use ANOVA to select best features
feature_selection = SelectKBest(score_func=f_classif,k=11)

#Feature Extractoin
#create a pipeline with filter based feature selection
pipe= make_pipeline(power_transformer,feature_selection,log_reg_model)
pipe.fit(X_train_poly,y_train)
y_prediction= pipe.predict(X_test_poly)
print("Accuracy of 2 degree polynomial Logistic Regression(after Yoe-Johnson transformation and selecting 11 features using ANOVA):",accuracy_score(y_test,y_prediction))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,2)))
print("Confusion matrix:",confusion_matrix_df.head())
print("-"*25,"logistic regression Metrics","-"*25)
print("Precision:  ",precision_score(y_test,y_prediction))
print("Recall: ",recall_score(y_test,y_prediction))
print("F1 score: ",f1_score(y_test,y_prediction))
print("-"*80)
print(classification_report(y_test,y_prediction))
#Create the feature selection report for all the features
