#Predicting Student Mental Health Using Machine Learning Models

#1. Introduction
##Objective: 
The goal of this project is to predict whether a student is at risk of mental illness based on a range of features including academic performance, social and family support, financial stress, lifestyle habits, and other factors. By building predictive models using Logistic Regression models, this project aims to provide insights that could potentially inform early interventions for mental health support in educational institutions.
##Background and Motivation:
Mental health challenges among students are increasingly prevalent due to various pressures from academic, social, and financial aspects of student life. By developing a data-driven approach, this project hopes to identify students at a higher risk of mental health issues. A predictive model that flags these risks can serve as a supplementary tool for mental health professionals in educational institutions, allowing them to focus support efforts on students who may be in need.

#2. Data Collection
##Data Source: ChatGPT
##Dataset Description:  The dataset comprises 300 entries, with each entry containing features related to a student’s demographics, lifestyle, and academic experience. The key features include:
•	Age
•	Gender
•	Academic Performance
•	Financial Stress (e.g., facing financial difficulties)
•	Family Support (rated on a scale)
•	Peer Support (rated on a scale)
•	Exercise Frequency (days per week)
•	Hours of Sleep
•	Screen Time Hours
•	Substance Use
•	Academic Pressure (self-reported on a scale)
•	Social Media Use
•	Access to Mental Health Support
##Target variable:  Diagnosed Mental Illness (binary value), indicates whether a student has been identified with a mental health condition.

#3. Exploratory Data Analysis (EDA)	
##Summary Statistics: Mean, median, and distribution of each feature.

##Visualizations:
- Histograms to assess the distribution of each variable. 
- Correlation bar graph to understand relationships between the various features and mental illness .

##Insights:  Accurate predictions about the likelihood of a student being diagnosed with mental illness. 

#4. Data Preprocessing
##Encoding Categorical Variables: One-hot encoding for any categorical features.

##Feature Scaling: The dataset contains features with different scales, so standardization is applied to ensure consistency, particularly for models sensitive to feature scales.

#5. Machine Learning Model Selection
##Model Choices:
- Neural Network Classifier (or handling high-dimensional data to capture complex dependencies among features).
- Random Forest Classifier (can capture complex interactions among features).
- Support Vector Machine (SVM) can handle data through kernel transformations.

##Why Scikit-Learn: Easy implementation, variety of algorithms, and effective performance metrics.

##Evaluation Metric:
- Accuracy: This metric assesses the overall rate of correct predictions across the test set.
- Confusion Matrix: Offers insight to analyze model performance in identifying high-risk students.

#6. Model Implementation
##Data Splitting: Split dataset into 80% training and 20% testing sets using `train_test_split` from Scikit-Learn.

##Hyperparameter Tuning:
Grid search or randomized search can be applied for SVM (e.g., adjusting kernel type and regularization), Random Forest (e.g., tuning the number of trees and depth), and Neural Network (e.g., optimizing layer sizes and learning rates).

##Code Example:
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score,accuracy_score

## Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)

## To display the Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred
)
## To find the accuracy score and f1 score of the predicted dataset
accuracy_score(y_test, y_pred)
f1_score(y_test, y_pred)

## Hyperparameter tuning for Neural Network Classifier
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # First hidden layer with 16 neurons
        self.fc2 = nn.Linear(16, 8)            # Second hidden layer with 8 neurons
        self.fc3 = nn.Linear(8, 2)             # Output layer with 2 classes
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## Hyperparameter tuning for Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

## Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

#7. Results and Evaluation
##Model Performance:
- Neural network classifier achieved a loss of 0.0001 indicating the model’s strength in predicting in mental illness accurately.
-Similarly, Random Forest Classifier and SVM plot the graphs indicating the accuracy in predicting mental illness accurately.

##Feature Importance:
- Insights into which factors (e.g., family support, financial stress) are most predictive of mental health issues, guiding the development of targeted interventions.
##Confusion Matrix: Visualized true vs. predicted values to identify common misclassifications.

#8. Conclusion and Future Work
Machine learning models effectively predict the likelihood of a student being diagnosed with mental illness, with an emphasis on how much each feature contribute to the development of mental illness. The project demonstrates potential threats for mental illness in students.

##Applications & Future Work:
Support the prioritization of mental health resources in educational settings.
Inform policies on improving student wellness based on data-backed findings.
Guide future research on mental health predictors within student populations.

#9. References
- Scikit-Learn Documentation
- ChatGPT










