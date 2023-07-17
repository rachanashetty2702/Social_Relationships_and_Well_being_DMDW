import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
import warnings

# Suppress UserWarning messages
warnings.filterwarnings("ignore", category=UserWarning)

target_variable = '16) How would you rate your own social skills in interpersonal interactions?'

# Load the Dataset
data = pd.read_csv('dataset/preprocessed_dataset.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Split the Data into Training and Testing Sets
X = data.drop(target_variable, axis=1)  # Replace 'target_variable' with the name of your target variable column
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a Classification Model
model = DecisionTreeClassifier()

# Train the Model
model.fit(X_train, y_train)

#Visualize the Decision Tree
# fig, ax = plt.subplots(figsize=(10, 15))  # Increase the size of the figure
# plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_.astype(str).tolist(), fontsize=8, ax=ax)
# plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9)  # Adjust the spacing between nodes
# plt.savefig("decision_tree.png", format="png", dpi=300)
# plt.show()


# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Performance and Construct Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'
f_measure = f1_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F-Measure:", f_measure)
print(y_pred)

def prediction():
    # Define the Mapping for the Selected Questions
    age_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    gender_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    marital_status_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    social_network_size_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    family_support_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    loneliness_levels_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    social_support_availability_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    emotional_wellbeing_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    social_interaction_frequency_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    social_connectedness_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    perceived_social_support_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    life_satisfaction_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    trust_in_relationships_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    supportive_friends_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    community_involvement_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4}


    # Form Questions and Collect User Input
    age = input("1] Age:\nWhat is your age?\na) 18-25\nb) 26-35\nc) 36-45\nd) 46 and above\n")
    gender = input("2] Gender:\nHow do you identify your gender?\na) Male\nb) Female\nc) Non-binary\nd) Prefer not to say\n")
    marital_status = input("3] Marital Status:\nWhat is your marital status?\na) Single\nb) Married\nc) Divorced\nd) In a committed relationship\n")
    social_network_size = input("4] Social Network Size:\nHow would you describe the size of your social network?\na) Small (fewer than 10 people)\nb) Moderate (10-50 people)\nc) Large (51-100 people)\nd) Very large (more than 100 people)\n")
    family_support = input("5] Family Support:\nHow would you rate the level of support and closeness you feel from your family members?\na) Low\nb) Moderate\nc) High\nd) Very high\n")
    loneliness_levels = input("6] Loneliness Levels:\nTo what extent do you experience feelings of loneliness?\na) Rarely or never\nb) Occasionally\nc) Sometimes\nd) Frequently\n")
    social_support_availability = input("7] Social Support Availability:\nHow readily available is social support for you in times of need?\na) Not available at all\nb) Somewhat available\nc) Moderately available\nd) Readily available\n")
    emotional_wellbeing = input("8] Emotional Well-being:\nHow would you rate your overall emotional well-being and happiness levels?\na) Very low\nb) Low\nc) Moderate\nd) High\n")
    social_interaction_frequency = input("9] Social Interaction Frequency:\nHow often do you engage in social interactions with friends, family, or acquaintances?\na) Daily\nb) Several times a week\nc) Once a week\nd) Rarely\n")
    social_connectedness = input("10] Social Connectedness:\nTo what extent do you feel a sense of belongingness and connectedness to others?\na) Not at all\nb) Somewhat\nc) Moderately\nd) Strongly\n")
    perceived_social_support = input("11] Perceived Social Support:\nHow would you rate the support you receive from others?\na) Very low\nb) Low\nc) Moderate\nd) High\n")
    life_satisfaction = input("12] Life Satisfaction:\nOverall, how satisfied are you with your life circumstances?\na) Very dissatisfied\nb) Dissatisfied\nc) Satisfied\nd) Very satisfied\n")
    trust_in_relationships = input("13] Trust in Relationships:\nHow much trust do you have in your close relationships?\na) Very little\nb) Some\nc) A moderate amount\nd) A great deal\n")
    supportive_friends = input("14] Supportive Friends:\nHow supportive are your friends in times of need?\na) Not supportive\nb) Somewhat supportive\nc) Moderately supportive\nd) Extremely supportive\n")
    community_involvement = input("15] Community Involvement:\nHow involved are you in community activities or organizations?\na) Not involved at all\nb) Minimally involved\nc) Moderately involved\nd) Actively involved\n")

    # Map the User Input to Numerical Values
    age = age_mapping[age]
    gender = gender_mapping[gender]
    marital_status = marital_status_mapping[marital_status]
    social_network_size = social_network_size_mapping[social_network_size]
    family_support = family_support_mapping[family_support]
    loneliness_levels = loneliness_levels_mapping[loneliness_levels]
    social_support_availability = social_support_availability_mapping[social_support_availability]
    emotional_wellbeing = emotional_wellbeing_mapping[emotional_wellbeing]
    social_interaction_frequency = social_interaction_frequency_mapping[social_interaction_frequency]
    social_connectedness = social_connectedness_mapping[social_connectedness]
    perceived_social_support = perceived_social_support_mapping[perceived_social_support]
    life_satisfaction = life_satisfaction_mapping[life_satisfaction]
    trust_in_relationships = trust_in_relationships_mapping[trust_in_relationships]
    supportive_friends = supportive_friends_mapping[supportive_friends]
    community_involvement = community_involvement_mapping[community_involvement]

    # Prepare the User Input as a NumPy Array
    user_input = [age, gender, marital_status,social_network_size, family_support, loneliness_levels, social_support_availability,
              emotional_wellbeing, social_interaction_frequency, social_connectedness, perceived_social_support,
              life_satisfaction, trust_in_relationships, supportive_friends, community_involvement]

    # Reshape the New Student Data
    new_student = np.array(user_input).reshape(1, -1)  # Convert the data to a 2D array

    # Make Predictions
    prediction = model.predict(new_student)

    # Print the Predicted Result
    if prediction == 1:
        return "Average"
    elif prediction == 2:
        return "Below Average"
    elif prediction == 0:
        return "Above Average"
    elif prediction == 3:
        return "Very Poor"

print(prediction())