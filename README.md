# Feature_Importance

In this repository I will explore different feature importance and selection methods. 
The feature importance techniques I explore in this repository are:
1. Spearman's rank correlation coefficient
2. Principle Component Analysis
3. Permutation Importance
4. Drop Column Importance
5. Shap Importance

After that, I train random forest models using 1,2,3.. most important features, using the techniques mentioned above, and then compare the loss to identify which technique is working best. 
I also implemented a simple Automatic Feature Selection function which iteratively drops the least important feature given by the feature importance method until it finds the best model (stops dropping when loss starts to get worse; there is also a grace range for how much the loss can get worse before the functions stops dropping).

Next I implemented functions to do a statistical analysis on the feature importance methods. These functions/statistical techinques are:
1. Variance in feature importance
2. Empirical P-Values
