import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# 1A. Basic visualization

plt.figure(figsize=(7, 5))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y
)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

plt.colorbar(label="Flower Class")

plt.tight_layout()

plt.savefig(
    "outputs/sepal_scatter.png"
)

plt.show()

# 1B. Petal visualization

plt.figure(figsize=(7, 5))

plt.scatter(
    X.iloc[:, 2],
    X.iloc[:, 3],
    c=y
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.colorbar(label="Flower Class")

plt.tight_layout()

plt.savefig(
    "outputs/petal_scatter.png"
)

plt.show()


# 2. Display dataset
print("AI DATA CLASSIFICATION PROJECT - classification.py:72")
print("" * 40)

print("\nDataset: - classification.py:75")
print(X.head())

print("\nDataset shape: - classification.py:78")
print(X.shape)

print("\nClasses: - classification.py:81")
print(iris.target_names)


# 3. Check missing values
print("\nMissing values: - classification.py:86")
print(X.isnull().sum())


# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples: - classification.py:98", len(X_train))
print("Testing samples: - classification.py:99", len(X_test))


# 5. Create classification model
model = DecisionTreeClassifier(
    random_state=42
)


# 6. Train model
model.fit(
    X_train,
    y_train
)

print("\nModel training completed! - classification.py:114")


# 7. Make predictions
predictions = model.predict(X_test)


# 8. Calculate accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy: - classification.py:127")
print(round(accuracy * 100, 2), "% - classification.py:128")


# 9. Classification report
print("\nClassification Report: - classification.py:132")
print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)


# 10. Confusion matrix
matrix = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix: - classification.py:148")
print(matrix)


# 11. Visualize confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(matrix)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.colorbar()

plt.xticks(
    [0, 1, 2],
    iris.target_names
)

plt.yticks(
    [0, 1, 2],
    iris.target_names
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            matrix[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()


# 12. Test new data
new_flower = [[
    5.1,
    3.5,
    1.4,
    0.2
]]

prediction = model.predict(
    new_flower
)

print("\nNew Flower Prediction: - classification.py:204")
print(
    iris.target_names[prediction[0]]
)


# 13. Finish
print("\n - classification.py:211" + "=" * 40)
print("PROJECT COMPLETED SUCCESSFULLY! - classification.py:212")
print("= - classification.py:213" * 40)

# 13. Compare multiple models

logistic_model = LogisticRegression(max_iter=200)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

logistic_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

random_forest_prediction = random_forest_model.predict(X_test)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)

print("\nModel Comparison - classification.py:242")
print("" * 40)

print(
    "Decision Tree:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Logistic Regression:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(random_forest_accuracy * 100, 2),
    "%"
)

models = [
    "Decision Tree",
    "Logistic Regression",
    "Random Forest"
]

scores = [
    accuracy,
    logistic_accuracy,
    random_forest_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(
    models,
    scores
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.show()

# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n - classification.py:300" + "=" * 50)
print("MODEL COMPARISON - classification.py:301")
print("= - classification.py:302" * 50)


# Logistic Regression
logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(
    X_train,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)


# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_prediction = random_forest_model.predict(
    X_test
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_prediction
)


# Display results
print("\nDecision Tree Accuracy: - classification.py:345")
print(round(accuracy * 100, 2), "% - classification.py:346")

print("\nLogistic Regression Accuracy: - classification.py:348")
print(round(logistic_accuracy * 100, 2), "% - classification.py:349")

print("\nRandom Forest Accuracy: - classification.py:351")
print(round(random_forest_accuracy * 100, 2), "% - classification.py:352")


# ============================================================
# FIND BEST MODEL
# ============================================================

model_names = [
    "Decision Tree",
    "Logistic Regression",
    "Random Forest"
]

model_scores = [
    accuracy,
    logistic_accuracy,
    random_forest_accuracy
]

best_index = model_scores.index(
    max(model_scores)
)

best_model = model_names[best_index]

best_score = model_scores[best_index]

print("\n - classification.py:379" + "=" * 50)
print("BEST MODEL - classification.py:380")
print("= - classification.py:381" * 50)

print("Best Model: - classification.py:383", best_model)

print(
    "Best Accuracy:",
    round(best_score * 100, 2),
    "%"
)


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    model_names,
    model_scores
)

plt.title("Machine Learning Model Comparison")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.show()