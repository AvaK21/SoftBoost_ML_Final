# SoftBoost
- - 
The project started with the base model of AdaBoost. The extension I implemented was to mesh AdaBoost and SVM loss functionality inspired by Rätsch’s Soft Margins for AdaBoost, with quadratic loss with slack variables similar to support vector machines (2001). As he mentions, AdaBoost uses a hard margin pattern for predicting samples. AdaBoost does not perform well when there is noise in the data or overlapping class distributions because it chases outliers and overfits to them (2001). The goal is to improve the Adaboost model's robustness when it interacts with overlapping class distributions and noise. Therefore, replace hard margins with soft margins to allow misclassifications in the margins and to reduce overfitting to abnormal data points.  


## How to run
pip install -r requirements.txt
python final_complete.py

**Expected**
Information about the Dataset and the number results are printed in the terminal. AdaBoost Model will have 2 images: Confusion matrix, plot of the weights' evolution. The SoftBoost will have an additional image of plot of the loss evolution.

## Dataset
Titanic Dataset from Kaggle
[Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

The dataset used was the Titanic Dataset on Kaggle[2]. The dataset has 891 samples (people) out of the 2224 people on the Titanic in 1912 when the ship sank. The target feature to predict is whether the individual survived the Titanic. The other features provided include passenger ID, ticket class (1st, 2nd, 3rd class), name, gender, age, number of siblings/spouses on board, number of parents/children on the Titanic, ticket number, passenger fare, cabin number, and embarked (what port they entered the Titanic on). 
Some of these features are not useful in prediction, and some have some missing values. So data cleaning needs to occur after loading the dataset and before giving it to the models. I looked at what others have done with this dataset to prepare it for prediction models. A Python notebook by babyxkush, helped with cleaning valuable features with missing data for some people: age and embark[3]. I removed the Cabin column, 77% missing values. Removed other columns with unique values for each person: passenger ID, name, and ticket number. Embarked had 2 missing values. Over 70% of the people had an embarked value of ‘S’ = Southampton. I set the 2 missing elements to be S. Age was missing in 20% of samples. However, age is very important in determining if someone survived the Titanic. Solution: Find samples with the same gender and ticket class and give the missing sample the median age of the group. 
Further cleaning of the data included one-hot-encoding the embarked and gender columns so the model can more easily interpret patterns[5]. Last but not least was changing the y (target) numpy array from {0,1} to  {-1,1} to conform to AdaBoost standards. The Titanic dataset was selected because the target feature is not 50% yes and no, and has an overlapping class distribution. This can be difficult for the vanilla AdaBoost model.

## From Scratch

From previous homework assignments I imported AdaBoost and Decision Stump classes. 
I imported numpy, pandas, matplotlib.pyplot, ConfusionMatrixDisplay and train_test_split from sklearn.

I took the AdaBoost class model and modified the calculate weights and update step in the fit function. The calculated loss and weights were the combination of hinge and exponential functions. They used a lam variable to be the ratio of SVM (hinge) and AdaBoost (exponential) implementation is used on update step on that instance of SoftBoost. lam = 0 is pure SVM, lam 1.0 is pure AdaBoost, and 0.5 is split evenly between them. 

In the update step, an ensemble score was added. The ensemble score comes from SVM. THe sign states what the prediction is {+1,-1} and the magnitude conveys how confident the prediction is. The margin is the correct classification times ensemble score. The ensemble score is used determine the predictions in the predict function.

For evalutaion and analysis I implemented: accuracy, precision, f1 score, precision, plot weight evolution, plot loss history, and load clean data. These functions were used in my main function to illustrate the findings of how the models did. In main, I made an instance of the classic AdaBoost and SoftBoost models with lam values of [0.0,0.1,0.5,0.9,1.0].

## Results Discussion



The AdaBoost weight evolution plot showed that the graph was overfitting to overlapping class samples, with a few samples having high weights while the rest had very little weight and flat lines. 

The worst SoftBoost was lam = 0.0 and 0.1. They had different epsilons and alphas, but the results measured were all the same. The training accuracy was 78.79%, and the testing accuracy of 78.21%. This is when only hinge loss is utilized or exponential loss has very little impact. It makes sense that a (mostly) pure SVM loss functionality doesn’t work well with an AdaBoost base model. The weights over time for lam = 0.0 exploded in magnitude. Both have a sawtooth graph indicating underfitting. 

The best SoftBoost was lam = 0.9. Training accuracy of 82.02%. Testing accuracy of 82.68%. Precision values of 81.16%. Recall is 75.68%. F1 score is 78.32%. Lam = 0.9 beat regular AdaBoost on every measurement for the tested data. The weights evolution graph shows an increase of samples with higher weights in later boosting rounds. But it still chases some overlapping class samples. At lam = 0.9, the model is mostly AdaBoost with a little hinge influence, to allow a few misclassifiers in the margin and soft margins.

At lam = 1.0, the model is a vanilla AdaBoost model, verified by having the same values and graphs as the regular AdaBoost model.

At lam = 0.5, the model is half AdaBoost and half SVM loss function. It had the highest training accuracy of 82.44%, but about -3% on testing accuracy with 79.33%. The weight's evolution showed more samples had an impact on the next decision stump than other models.

Overall, the SoftBooster helped with generalization, when it uses mostly an AdaBoost loss with slight influence from SVM to allow softer margins. Comparing lam = 0.9 to AdaBoost, there an increase in testing accuracy 0.56%, precision 0.28%, recall 1.26%, and F1 score 0.86%.
	
I was surprised that lam = 0.5 didn’t do as well or better than the regular AdaBoost. Also, I was shocked that recall, precision, and F1 score were in the same range of high 60s to low 80s for all versions of the model. Usually they were in the low to mid-70s percentage.




## Learned
- Importance of cleaning data so the model can make useful predictions
- How you show your data is important
    - A picture or graph can convey more than a single number
    - Finding how show you data in graphs are important
- Better undtanding of the update step in machine learning models
- How different loss implementations can alter how well or poorly a model preforms

- - 
Note: Some of the sections of README.md are from my final paper. So the reference [#] can be found in my final paper.
 



add the .gitignore - virtual envinoment


if can theory
-soft margin boosting, loss function, weight update rule, softBoost is more robust to noisy labels, unserstand the math not just the code

