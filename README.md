BREAST CANCER CLASSIFICATION USING ENSEMBLE LEARNING

1. PROJECT TITLE

Breast Cancer Classification Using Ensemble Learning


2. PROJECT DESCRIPTION

This project demonstrates the application of different Ensemble
Learning techniques for Breast Cancer Classification.

Four major ensemble learning approaches are implemented and
studied:

1. Bagging
2. Random Forest
3. Boosting
4. Voting

Voting is further implemented using two approaches:

1. Hard Voting
2. Soft Voting

The objective of this project is to understand how ensemble learning
techniques combine multiple machine learning models or predictions
to improve classification performance.


3. PROBLEM STATEMENT

The objective is to build machine learning classification models
that can classify breast cancer observations based on the given
diagnostic features.

The same dataset is used across different ensemble learning
techniques so that their working and performance can be studied
consistently.


4. MACHINE LEARNING TYPE

Learning Type:
Supervised Learning

Problem Type:
Classification

Domain:
Breast Cancer Classification

Dataset:
breast_cancer.csv

Target Column:
target


5. ENSEMBLE TECHNIQUES IMPLEMENTED

5.1 BAGGING

Algorithm:
BaggingClassifier

Base Estimator:
Decision Tree Classifier

Number of Estimators:
10

Bagging creates multiple instances of the base model using
different bootstrap samples of the training data.

The predictions of the individual models are combined to generate
the final prediction.

Concept:

Training Dataset
       |
       +---- Decision Tree 1
       |
       +---- Decision Tree 2
       |
       +---- Decision Tree 3
       |
       +---- ...
       |
       +---- Decision Tree 10
       |
       v
Combine Predictions
       |
       v
Final Prediction


5.2 RANDOM FOREST

Algorithm:
RandomForestClassifier

Number of Estimators:
10

Random Forest is an ensemble of multiple Decision Trees.

It introduces randomness in both the training samples and feature
selection during tree construction.

The predictions from multiple Decision Trees are combined to
generate the final classification result.

Concept:

Training Dataset
       |
       +---- Randomized Decision Tree 1
       |
       +---- Randomized Decision Tree 2
       |
       +---- Randomized Decision Tree 3
       |
       +---- ...
       |
       +---- Randomized Decision Tree 10
       |
       v
Majority Voting
       |
       v
Final Prediction


5.3 BOOSTING

Algorithm:
AdaBoostClassifier

Number of Estimators:
50

Learning Rate:
1.0

AdaBoost is a boosting algorithm that trains weak learners
sequentially.

Each new learner focuses more on the observations that were
incorrectly classified by previous learners.

The learners are then combined to produce the final prediction.

Concept:

Training Data
      |
      v
 Weak Learner 1
      |
      v
Focus on Incorrect Predictions
      |
      v
 Weak Learner 2
      |
      v
Focus on Incorrect Predictions
      |
      v
      ...
      |
      v
Final Weighted Prediction


5.4 HARD VOTING

Algorithm:
VotingClassifier

Voting Type:
Hard Voting

Models Used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors

Hard Voting uses the predicted class labels from the individual
models.

The class receiving the highest number of votes becomes the final
prediction.

Example:

Logistic Regression -> Benign
Decision Tree       -> Malignant
KNN                 -> Benign

Benign     = 2 votes
Malignant  = 1 vote

Final Prediction = Benign


5.5 SOFT VOTING

Algorithm:
VotingClassifier

Voting Type:
Soft Voting

Models Used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors

Soft Voting uses the predicted probabilities from the individual
models.

The probabilities are combined and the class with the highest
combined probability becomes the final prediction.

Example:

Model 1:
Benign = 0.70
Malignant = 0.30

Model 2:
Benign = 0.60
Malignant = 0.40

Model 3:
Benign = 0.80
Malignant = 0.20

Average:

Benign = 0.70
Malignant = 0.30

Final Prediction = Benign


6. HETEROGENEOUS ENSEMBLE

The Voting models use different machine learning algorithms:

Logistic Regression
Decision Tree
KNN

Since different types of models are combined, Hard Voting and
Soft Voting in this project represent heterogeneous ensemble
learning.

The models use different learning strategies and their predictions
are combined to generate the final result.


7. HOMOGENEOUS ENSEMBLE

Bagging and Random Forest in this project primarily use Decision
Trees as their base learning models.

Therefore, multiple models of the same general model family are
combined.

This represents a homogeneous ensemble approach.


8. DATASET

Dataset File:

breast_cancer.csv

The dataset contains multiple diagnostic features.

Target Column:

target

The target column is separated from the remaining feature columns.

X:
Independent variables / Features

Y:
Dependent variable / Target


9. COMMON PROJECT WORKFLOW

All classification programs generally follow these machine learning
steps:

Step 1:
Load the dataset.

Step 2:
Separate independent variables and target variable.

Step 3:
Split the dataset into training and testing data.

Step 4:
Perform feature scaling where applicable.

Step 5:
Create the machine learning model.

Step 6:
Train the model.

Step 7:
Generate predictions.

Step 8:
Evaluate model performance.


10. DATA PREPROCESSING

The following preprocessing techniques are used:

Dataset loading using pandas.

Separation of features and target.

Train-test split.

Feature scaling using StandardScaler where applicable.


11. MODEL EVALUATION

The models are evaluated using:

1. Accuracy Score

Accuracy measures the percentage of correctly classified
observations.

2. Confusion Matrix

The confusion matrix provides a detailed view of correct and
incorrect classification predictions.


12. LIBRARIES USED

pandas

Used for dataset loading and data manipulation.

scikit-learn

Used for machine learning algorithms, preprocessing, ensemble
learning, dataset splitting and evaluation.


13. PROJECT STRUCTURE

Breast-Cancer-Ensemble-Learning/
|
+-- README.txt
|
+-- requirements.txt
|
+-- .gitignore
|
+-- data/
|   |
|   +-- breast_cancer.csv
|
+-- bagging/
|   |
|   +-- breast_cancer_bagging.py
|
+-- random_forest/
|   |
|   +-- breast_cancer_random_forest.py
|
+-- boosting/
|   |
|   +-- breast_cancer_boosting.py
|
+-- voting/
|   |
|   +-- breast_cancer_voting_hard.py
|   |
|   +-- breast_cancer_voting_soft.py
|
+-- results/
    |
    +-- model_comparison.txt


14. SOFTWARE REQUIREMENTS

Operating System:

Windows / Linux / macOS

Programming Language:

Python 3.10 or later

Development Environment:

Visual Studio Code


15. INSTALLATION

Step 1:

Install Python 3.10 or later.

Step 2:

Clone or download this repository.

Step 3:

Open the project in Visual Studio Code.

Step 4:

Open the terminal in the project directory.

Step 5:

Install the required libraries:

pip install -r requirements.txt


16. RUNNING THE PROJECT

Bagging:

python bagging/breast_cancer_bagging.py


Random Forest:

python random_forest/breast_cancer_random_forest.py


Boosting:

python boosting/breast_cancer_boosting.py


Hard Voting:

python voting/breast_cancer_voting_hard.py


Soft Voting:

python voting/breast_cancer_voting_soft.py


17. COMPARISON OF ENSEMBLE TECHNIQUES

Bagging:

Main idea:
Train multiple models independently using bootstrap samples.

Base Model:
Decision Tree

Training:
Parallel / Independent

Combination:
Voting for classification


Random Forest:

Main idea:
Build multiple randomized Decision Trees.

Base Model:
Decision Tree

Training:
Multiple randomized trees

Combination:
Majority voting


Boosting:

Main idea:
Train models sequentially and focus on previous errors.

Algorithm:
AdaBoost

Training:
Sequential

Combination:
Weighted contribution of learners


Hard Voting:

Main idea:
Combine class predictions from different models.

Models:
Logistic Regression
Decision Tree
KNN

Combination:
Majority vote


Soft Voting:

Main idea:
Combine class probabilities from different models.

Models:
Logistic Regression
Decision Tree
KNN

Combination:
Probability aggregation


18. KEY CONCEPTS DEMONSTRATED

Supervised Learning

Classification

Train-Test Split

Feature Scaling

Decision Tree

Bagging

Random Forest

Boosting

AdaBoost

Voting Classifier

Hard Voting

Soft Voting

Homogeneous Ensemble

Heterogeneous Ensemble

Majority Voting

Probability Aggregation

Accuracy

Confusion Matrix


19. IMPORTANT DIFFERENCE BETWEEN THE APPROACHES

Bagging:

Multiple models are trained independently using different
bootstrap samples.

Random Forest:

Multiple randomized Decision Trees are created using bootstrap
samples and random feature selection.

Boosting:

Models are trained sequentially and later models focus more on
previously misclassified observations.

Hard Voting:

Final prediction is based on the majority of predicted class
labels.

Soft Voting:

Final prediction is based on combined class probabilities.


20. LEARNING OBJECTIVE

The main purpose of this project is to gain practical understanding
of Ensemble Learning and compare different ensemble techniques on
the same classification problem.

The project demonstrates both:

Homogeneous Ensemble Learning

and

Heterogeneous Ensemble Learning.


21. AUTHOR

Pratiksha Gorakh Mahale