import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
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
#split data in training and test set
X_train,X_test,y_train,y_test = train_test_split(heart_df.drop(columns=['target']),
                                               heart_df['target'],test_size=0.2,random_state=20)
#standard scaler
std_scaler= StandardScaler().set_output(transform="pandas")
#scale the data
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

#logistic regression
log_reg_model = LogisticRegression()
#Feature Selection:
# Filter based method: We'll use ANOVA to select best features
feature_selection = SelectKBest(score_func=f_classif,k=8)
#wrapper method
#exhaustive feature selection
wrapper_exhaustive_FS=EFS(log_reg_model, max_features=11,
			  scoring='accuracy',
			  cv=5,
			  print_progress=True,	#displays the whole process
			  n_jobs=-1)
#Sequential forward selection
wrapper_SFS_FS=SFS(log_reg_model, k_features='best', #we are not specifying any number of features instead asking sklearn to tell us the best number
			forward=True,
			floating=False,
			scoring='accuracy',
			cv=5)
#Sequential backward elimination
wrapper_SBE_FS=SFS(log_reg_model, k_features='best', #we are not specifying any number of features instead asking sklearn to tell us the best number
			forward=False,
			floating=False,
			scoring='accuracy',
			cv=5)
#Feature Extractoin
print("Running Backward elimination feature selection")
SBE_model=wrapper_SBE_FS.fit(X_train_scaled,y_train)
#to get the subset of features with highest accuracy
print("Subset features (indices) with highest accuracy for Backward elimination:",SBE_model.k_feature_idx_)
print("Subset features (name) with highest accuracy for Backward elimination:",SBE_model.k_feature_names_)
print('Best accuracy score for Backward elimination: %.2f' % SBE_model.k_score_)
#plot the result
fig1 = plot_sfs(SBE_model.get_metric_dict(), kind='std_err',)
plt.title('Sequential Backward Elimination (w. StdErr)')
plt.grid()
plt.show()

print("Running Forward feature selection")
SFS_model=wrapper_SFS_FS.fit(X_train_scaled,y_train)
#to get the subset of features with highest accuracy
print("Subset features (indices) with highest accuracy for forward selection:",SFS_model.k_feature_idx_)
print("Subset features (name) with highest accuracy for forward selection:",SFS_model.k_feature_names_)
print('Best accuracy score: %.2f' % SFS_model.k_score_)
#plot the result
fig2 = plot_sfs(SFS_model.get_metric_dict(), kind='std_err',)
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

print("Running Exhaustive feature selection")
exhaustive_model= wrapper_exhaustive_FS.fit(X_train,y_train)
#to find the best score
print("Best scores:",wrapper_exhaustive_FS.best_score_)
#to find out which subset of features got the best score
print("Features:",wrapper_exhaustive_FS.best_feature_names_)
print('Best subset:', wrapper_exhaustive_FS.best_idx_)
