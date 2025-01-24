import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("D:\codealpha\Titanic-Dataset.csv" )
print(data.head())
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)

# Convert categorical data to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Extract features (X) and target (y)
X = data[features]
y = data[target]
X = X.fillna(0)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict survival for a new person
def predict_survival(pclass, sex, age, sibsp, parch, fare):
    sex = 0 if sex.lower() == 'male' else 1
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], 
                              columns=features)
    prediction = model.predict(input_data)[0]
    return "Survived" if prediction == 1 else "Did not survive"

# Example prediction
new_person = predict_survival(pclass=1, sex='female', age=28, sibsp=0, parch=0, fare=100)
print(f"Prediction for the new person: {new_person}")
