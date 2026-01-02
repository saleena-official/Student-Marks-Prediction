import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Load dataset
data = pd.read_csv("data/student_data.csv")

# Step 2: Features and target
X = data[['StudyHours', 'Attendance', 'SleepHours']]
y = data['Marks']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)

# Step 7: Evaluate
print("Actual Marks:", y_test.values)
print("Predicted Marks:", predictions)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))