#In this project we are going to classify different dataset using 
logistic regression. 
1. Data with 2 output groups: in binary_classifier.py we will try to 
implement classification on a data set(heart disease data that we 
downloaded kagel) that has two output variable.
1.1. We did EDA to identify the PDF of all the input variables and observed 
     different distribution for different variable. 
1.2. We decide to do Yeo-Johnson power transformer on input variables to normalize them.
1.3. We then checked the PDFs of all the input variables once again with normalized Data.
1.4. Finally, we made a heat map of all the features correlation with each other to identify which features
are most correlated.
2. We performed the logistic regression.
2.1. For normalizing the data, we used Yeo-Johnson transformer
2.2. For feature selection, we used fileter based (using ANOVA) technique. We observed, using 8 features we
got the best accuracy. The features are 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca' and 'thal'
3. We performed (in feature_selection.py) 3 types of wrapper based feature selection technique to figure which
features are producing optimal result. We observed
For Backward elimination:
Subset features (indices) with highest accuracy for Backward elimination: (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12)
Subset features (name) with highest accuracy for Backward elimination: ('sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal')
Best accuracy score for Backward elimination: 0.85
Running Forward feature selection
For Forward selection:
Subset features (indices) with highest accuracy for forward selection: (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
Subset features (name) with highest accuracy for forward selection: ('sex', 'cp', 'trestbps', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal')
Best accuracy score: 0.86
We observed, both "backward elimination" and "forward selection" identified subset of 11 features for the optimal result.
However, "backward elimination" selected 'chol' feature whereas "forward selection" selected "exang" as feature.
Based on our observation from backward elimination and forward selection we decided to try exhaustive search on 11 features.
Exhaustive feature selection with 11 features, produced 85.9% accuracy and selected features are 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca' and 'thal'.
4. So, we decided to run regression analysis using 'sex', 'cp', 'trestbps', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca' and 'thal' features. This time we observed,
83.9% accuracy from logistic regression.
5. Using the iris dataset, we apply softmax(iris_softmax.py).
#################### Softmax on iris data ########################################
#EDA
Using ProfileReport we did EDA analysis on the iris data (see iris_output.html)
#Pre-processing
1. We observed duplicated rows, therefore removed them.
2. We scaled the data and normalized it before running softmax.
#################### polynomial logistic regression ########################################
From EDA step, it was evident that the heard data didn't have linear relationship. Therefore,
we also applied polynomial logistic regression on the heart data set to see if we get better result. 
But it produced only 80.9% accuracy (best score with 3 degree polynomial factor)