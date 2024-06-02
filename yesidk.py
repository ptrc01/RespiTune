# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Generate dummy data
np.random.seed(42)

# Generate interview questions data with higher values indicating higher stress tolerance
interview_questions = np.random.randint(1, 50, 100)

# Generate years of experience data
years_experience = np.random.randint(1, 25, 100)

# Generate hours willing to work data
hours_willing_to_work = np.random.randint(1, 10, 100)  # Adjusted to 100 data points

# Create a DataFrame to hold the data
data = pd.DataFrame({
    'Interview_Questions': interview_questions,
    'Years_Experience': years_experience,
    'Hours_Willing_To_Work': hours_willing_to_work
})

# Assign stress level tolerance based on the data
# The higher the value of interview questions, years of experience, and hours willing to work, the stronger the stress tolerance
# Weights: interview questions > years of experience > hours willing to work

# Create a weighted score
weighted_score = (data['Interview_Questions'] * 0.5) + (data['Years_Experience'] * 0.3) + (data['Hours_Willing_To_Work'] * 0.1)

# Categorize stress levels based on the weighted score
data['Stress_Level'] = pd.cut(weighted_score, bins=[0, 10, 15, 25, 50, np.inf], labels=[0, 1, 2, 3, 4], right=False).astype(int)

# Prepare data for training
X = data.drop(columns=['Stress_Level'])
y = data['Stress_Level']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Support Vector Machine (SVM) classifier
model = SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Function to predict stress level for a single client
def predict_stress_level(interview_questions, years_experience, hours_willing_to_work):
    # Create a DataFrame with the input data
    new_data = pd.DataFrame({
        'Interview_Questions': [interview_questions],
        'Years_Experience': [years_experience],
        'Hours_Willing_To_Work': [hours_willing_to_work]
    })
    
    # Standardize new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    
    # Return the predicted stress level
    return prediction[0]

# Example usage:
predicted_stress_level = predict_stress_level(17, 4, 7)
print("Predicted Stress Level:", predicted_stress_level)
