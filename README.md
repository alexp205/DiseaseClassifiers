# DiseaseClassifiers
A small collection of classification and regression decision tree and random forest-based projects designed for practice and exploration of machine learning.

The over-arching project is split into a few general components:
1. Decision Tree Classifiers - a decision tree is trained on heart disease data and must predict the "likelihood" of heart disease in the patient
2. Decision Tree Regression - similar to 1, but now a different, continuous value is being predicted
3. Random Forest Classifiers - a look into how random forests work in general as well as in relation to decision trees, this specific application focuses on classification
4. Random Forest Regression - similar to 3, but with regard to regression

NOTE: Regression and error-based pruning have not yet been implemented. They will be future improvements to the project.

## Data Information
The dataset was pulled from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/index.php). Specifically, the heart disease dataset (http://archive.ics.uci.edu/ml/datasets/Heart+Disease) was used due to the amount of available data, the variety of data types, and its applicability.

The specific details for each project are as follows:
* Decision Tree Classifiers
    * Data ordering for the discrete dataset (according to the UCI attribute documentation) (I think):
        1. #4 (sex)
        2. #9 (cp)
        3. #13 (smoke)
        4. #19 (restecg)
        5. #38 (exang)
        6. #41 (slope)
        7. #44 (ca)
        8. #51 (thal)
    * Data ordering for the continous dataset (according to the UCI attribute documentation) (I think):
        1. #3 (age) 
        2. #4 (sex) 
        3. #9 (cp) 
        4. #10 (trestbps) 
        5. #12 (chol) 
        6. #16 (fbs) 
        7. #19 (restecg) 
        8. #32 (thalach) 
        9. #38 (exang) 
        10. #40 (oldpeak) 
        11. #41 (slope) 
        12. #44 (ca) 
        13. #51 (thal)
    * The predicted value was #58, that is the probability of heart disease presence in the patient.

The random forests use the same data format.

## Structure Notes
A few notes on the chosen structure:
 - the splitting algorithm used at nodes calculates entropy and maximum information gain
 - decision tree anti-overfitting relies on cutting off expansion prematurely, specifically, the program takes the square root of the input data size as the minimun number of data points before that node is turned into a leaf by majority vote labeling
 - if the random forest size and bagging size are not specified, the defaults are (respectively) 1000 and the input data size divided by 5 (with a minimum of 10)
 - the random sampling of data at the tree nodes in the random forest take data (without replacement) until the square root of the input data size (rounded up) is reached

## USAGE:
1. [string] the path to the training data csv file
2. [string] the path to the testing data csv file
3. [bool] determines whether or not the data is discrete or continuous
4. [bool] determines whether or not the task is classification or regression (i.e. discrete, finite or continous labels)
5. [bool] determines whether or not to use a random forest
6. [int] number of trees in random forest
7. [int] bagging size of tree data in random forest
