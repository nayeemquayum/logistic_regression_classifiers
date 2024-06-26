import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
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
#logistic regression
log_reg_model = LogisticRegression()
#Feature Selection:
# Filter based method: We'll use ANOVA to select best features
feature_selection = SelectKBest(score_func=f_classif,k=11)

#Feature Extractoin
#create a pipeline with filter based feature selection
pipe= make_pipeline(power_transformer,feature_selection,log_reg_model)
pipe.fit(X_train,y_train)
y_prediction= pipe.predict(X_test)
print("Accuracy of Logistic Regression(after Yoe-Johnson transformation and selecting 11 features using ANOVA):",accuracy_score(y_test,y_prediction))
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,2)))
print("Confusion matrix:",confusion_matrix_df.head())
print("-"*25,"logistic regression Metrics","-"*25)
print("Precision:  ",precision_score(y_test,y_prediction))
print("Recall: ",recall_score(y_test,y_prediction))
print("F1 score: ",f1_score(y_test,y_prediction))
print("-"*80)
print(classification_report(y_test,y_prediction))
#Create the feature selection report for all the features
# display selected feature names
print("Top selected features",X_train.columns[feature_selection.get_support()])
feature_scores_df = pd.DataFrame(feature_selection.scores_)
features_name_df = pd.DataFrame(X_train.columns)
best_features_score_df = pd.concat([features_name_df,feature_scores_df],axis=1)
#rename the dataframe columns
best_features_score_df.columns = ['features','score']
print(best_features_score_df.sort_values('score',ascending=False))
#plot the result in 2D with top features 'sex' and 'cp'
heart_2D= heart_df.drop(['trestbps','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'], axis='columns')
#run logistic regression with top two important feature 'sex' and 'cp'
from mlxtend.plotting import plot_decision_regions
#split data in training and test set
X_2D_train,X_2D_test,y_2D_train,y_2D_test = train_test_split(heart_2D.drop(columns=['target']),
                                               heart_2D['target'],test_size=0.2,random_state=20)
log_reg_model.fit(X_2D_train,y_2D_train)
plot_decision_regions(X_2D_train.values, y_2D_train.values, log_reg_model, legend=2)

# Adding axes annotations
plt.xlabel('sex')
plt.xlabel('cp')
plt.title('logistic regression on heart')

plt.show()
