# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and split it into training and testing data.

2.Create and train the SGD Classifier using the training data.

3.Predict the Iris flower species using the trained model.

4.Evaluate the model using accuracy and classification metrics.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MENAKA M S
RegisterNumber: 212225040232 
*/
sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

## Output:

<img width="536" height="368" alt="image" src="https://github.com/user-attachments/assets/b67b4a3d-e92c-4598-b022-603ca2415929" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
