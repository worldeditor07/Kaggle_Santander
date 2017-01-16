# Kaggle_Santander

This is the project for Kaggle's Santander Customer Satisfaction Dataset. The link to the competition is https://www.kaggle.com/c/santander-customer-satisfaction

The data set has the following special structures:

#1. sparsity:
  96% of the data are zero, more than 90% of customers has more than 350/370 features completely zero. During the feature engineering, we found the numbers are records of various assets/financial products owned by customers and therefore we engineered a new feature which count the total quantitity/value of the assets. 
  
#2. highly-skewed dataset:
  The desired label of prediction is binary and out of 70,000 training samples, only 4.5% of customers are labeled as   unsatisfied. We found that truncated sigular valued decompostition and down sampling to be a good tool in this situation. In particular, TruncatedSVD is more appropriate than PCA since the data is sparse, and t-SVD does not require feature centralization as opposed to PCA, which destroies the sparsity.

#3. Categorical vs Numerical:
  All the features are presented as numerical data and we realized many of which can be simplified to categorical data. For example, the variable 'var3' has 95% of its value equal to 2 and the rest are distributed within [0, 100] and has around 1% of the samples being marked with -999999, we suspect this is a categorical variable and -999999 was originally labeled as 'NA'. 
  Another more general category of such variables are 'num_var40', 'num_var40_0', 'ind_var40' and so on, we found out many of them are colinear and only takes no more than 5 values, the corresponding categorical encoding is performed.
  
#4. Feature Selection:
  The features are understood to be assets owned by customers and during feature exploration, we found that customers who are richer/hold more assets/has higher balance/multiple transaction tend to be more likely to be unsatisfied. On the other side, customers who are in debt with the bank also tend to be unsatisfied while the majority of customers has zero records for holding/transferring balances. To amplify this intuition, we generated the following features:
  zero_sums: number of total zeros, this reflects the number of assets/activity of accounts 
  saldo_mean_ult/hace: the mean value of different assets in different time periods
  
  In practice, the Random Forest Classifier find these features, along with var3, var38, var15 to be among the top 10 most significant features in the model prediction. Even if we drop the other features relavent to 'saldo' or 'saldo_medio' variables, the performances remains the same and the model is simpler, more robust.
  

#5. Tree Models: 
By the above study of the features, we believe tree methods is more appropraite for the purpose and implemented XGBosst Classifier and Random Forest Classifier as the main model estimators. On the other hand, the highly skewed data implies that we should be more careful with the overfitting in training these models. In practice, we found downsampling enables RF to perform normally and L1-regularization significantly boost the performance of XGBoost models.

#6. Other Models:
It is still worthwhile to train other linear/non-linear models like SVM classifier, Logistic classifier or run clustering algorithms like KNN, DBScan, t-SNE to either learn more about the nature of data or for the purpose of getting more basic models/megafeatures for ensembling/bagging. In particular, a Neural Network model will be updated in later versions. It's reasonable to implement the convolutional neural network as we did observe patterns accoss the assets (as 1D-space).

  


  
