# Wines classifier
My project is divided in the following steps:

1. Exploration of the data;  
    - Statistical analysis of the data;  
    - Analysis of the features;  
    - Distribution of the features;      
2. Preparation of the data for modeling;    
    - Leading with categorical values;  
    - Leading with null and problematic features;  
3. Battle of models;  
    - Train the model with several classification algorithms (in this case: random forest and gradient boosting classifier);  
4. Evaluation of the models;  
    - Calculate the accuracy;  
    - Calculate the precision;  
    - Calculate the recall;  
    - Show the confusion matrix;  
5. Choose the best model;  
    - Choose the best model;  
    - Save the model as pickle.

## Exploration of the data
The exploration of the data shows the presence of a categorical feature that is 'type'. The type of wine could be red or white. Therefore I did a label encoder to this value to transform it in 0 or 1. The ML algorithms only lead with numerical values.

During the exploration I also found some mistakes in the feature 'alcohol'. Some values are not numeric, but strange strings of values. I decided to remove such lines from the data.

Furthermore I did an histogram for each feature and of the target to see the distribution. From the distributions I see the 'density' is very similar among the wines, however a little set of wines has a very different density. For that reason I train the model with and without such feature. 

Another fact is that, also if the classification could be between 0 and 10, in the dataframe the wines are classified only between 3 and 9. For that reason my classification algorithm could not classify a 0, 1, 2 or 10 wine.
## Prepare data for modeling
To prepare the data for modeling I did a label encoder in the 'type' feature. Furthermore, I removed the strange lines with the feature 'alcohol' equal to strange strings.
As I decided to use the random forest classification and the gradient boosting classification I did not need to do a standard scaler. Actually it is commonly known that for such algorithms the standard scalar is not necessary and it could get worse the result.
## Battle of models
I trained the model using two famous algorithms of classification:
- Random Forest Classification;
- Gradient Boosting Classification.
The random forest is faster than the gradient boosting, but normally the gradient boosting is more accurate. The best parameters for each algorithm are chosen with the grid search CV.
## Evaluation of the models
I evaluated the models calculating the accuracy, precision and recall and showing the confusion matrix.
### Results for the Random Forest Classification:
Accurancy: 0.69
Precision: [0., 0.67, 0.72, 0.67, 0.68, 0.95]  
Recall:    [0., 0.18, 0.73, 0.75, 0.61, 0.46]  
Precision (mean weighted): 0.69  
Recal (mean weighted): 0.69  
Confusion matrix:  
[[  0,   0,   5,   1,   0,   0],  
 [  0,   8,  22,  13,   1,   0],  
 [  0,   2, 312, 112,   2,   0],  
 [  0,   2,  89, 419,  48,   0],  
 [  0,   0,   7,  75, 132,   1],  
 [  0,   0,   0,  10,  12,  19]]  
### Results for the Gradient Boosting Classification:
Accurancy: 0.67  
Precision: [0., 0.47, 0.72, 0.65, 0.68,0.78]  
Recall:    [0., 0.20, 0.69, 0.77, 0.54,0.44]  
Precision (mean weighted): 0.67  
Recal (mean weighted): 0.67  
Confusion matrix:  
[[  0,   0,   6,   0,   0,   0],  
[  1,   9,  17,  14,   3,   0],  
[  0,   6, 295, 119,   8,   0],  
[  0,   4,  84, 432,  35,   3],  
[  0,   0,   9,  87, 117,   2],  
[  0,   0,   1,  12,  10,  18]]   

## Choose the best model
The evaluation of the model shows that the random forest classifier classify better the wines. Certainly, in a real case, with more time, I would have trained the model with a greater number of estimators. Finally, the model is saved as pickle and could be use to classify new wines.
