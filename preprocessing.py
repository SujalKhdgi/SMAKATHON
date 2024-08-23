import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load the dataset
    df = pd.read_csv(r'E:\SMAKATHON\original_dataset.csv')

    # Convert categorical variables to numerical variables
    df['Fever'] = df['Fever'].map({'Yes': 1, 'No': 0})
    df['Cough'] = df['Cough'].map({'Yes': 1, 'No': 0})
    df['Fatigue'] = df['Fatigue'].map({'Yes': 1, 'No': 0})
    df['Difficulty Breathing'] = df['Difficulty Breathing'].map({'Yes': 1, 'No': 0})

    # Convert categorical variable 'Disease' to numerical variables using LabelEncoder
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])

    # Create disease mapping after fitting the LabelEncoder
    disease_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    # Print the disease IDs and their corresponding names
    print("Disease IDs and their corresponding names:")
    for disease, id in disease_mapping.items():
        print(f"{id}: {disease}")

    # Convert categorical variable 'Gender' to numerical variables
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    # Convert categorical variables 'Blood Pressure' and 'Cholesterol Level' to numerical variables
    bp_mapping = {'High': 2, 'Normal': 1, 'Low': 0}
    cl_mapping = {'High': 2, 'Normal': 1, 'Low': 0}
    df['Blood Pressure'] = df['Blood Pressure'].map(bp_mapping)
    df['Cholesterol Level'] = df['Cholesterol Level'].map(cl_mapping)

    # Scale numerical variables using StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['Age']] = scaler.fit_transform(df[['Age']])

    #df.to_csv(r'E:\SMAKATHON\preprocessed_data.csv', index=False)

    # Return the preprocessed dataset
    return df

# Call the function to execute the preprocessing
preprocessed_df = preprocess_data()