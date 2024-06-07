import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline,make_pipeline	 #for pipeline
from sklearn.compose import ColumnTransformer
from ydata_profiling import ProfileReport

#load the data
iris_df = sns.load_dataset('iris')
print(iris_df.info())
#create profile report to understand the data
# prof = ProfileReport(iris_df)
# prof.to_file(output_file='iris_output.html')
#find duplicate rows
print("duplicate rows",iris_df.duplicated().sum())
#drop duplicated rows from dataframe
iris_df.drop_duplicates(inplace=True)
# sns.pairplot(iris_df, hue='species')
# plt.show()
encoder = LabelEncoder()
iris_df['species'] = encoder.fit_transform(iris_df['species'])
#split data in test and train
#split data in training and test set
X_train,X_test,y_train,y_test = train_test_split(iris_df.drop(columns=['species']),
                                               iris_df['species'],test_size=0.2,random_state=20)
std_scaler= ColumnTransformer([('standard_scaler',StandardScaler(),[0,1,2,3])
],remainder='passthrough')
#normalize the data using yeo-johnson power transformer
power_transformer= ColumnTransformer([
    ('yeo_john_trans',PowerTransformer(standardize=True,copy=False),[0,1,2,3])
],remainder='passthrough')
#logistic regression
softmax_model = LogisticRegression(multi_class='multinomial')
pipe= make_pipeline(std_scaler,power_transformer,softmax_model)
pipe.fit(X_train,y_train)
y_prediction= pipe.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy of Logistic Regression",accuracy_score(y_test,y_prediction))
#confusion_matrix
confusion_matrix=pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,3)))
print("Confusion matrix",confusion_matrix.head())
#recall_score,precision_score,f1_score
print("-"*25,"Softmax Metrics","-"*25)
print("Precision:  ",precision_score(y_test,y_prediction, average='weighted'))
print("Recall: ",recall_score(y_test,y_prediction, average='weighted'))
print("F1 score: ",f1_score(y_test,y_prediction, average='weighted'))
print("-"*80)