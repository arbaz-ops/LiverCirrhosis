from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os


app = Flask(__name__)


model_directory = "/Users/arbaz/Desktop/LiverCirrhosis"  # Path to the directory
model_path = os.path.join(model_directory, "xgboost_model.json")

# Load the model
loaded_booster = xgb.Booster()
loaded_booster.load_model(model_path)

print(loaded_booster)


df = pd.read_csv("/Users/arbaz/Desktop/liver_cirrhosis.csv")

df['Age'] = df['Age'] / 365
df['Age'] = df['Age'].round()

df['Age'] = df['Age'].astype(int)
df['Stage'] = df['Stage'].astype(int)


label_encoders = {}
categorical_features = ['Status', 'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema','Stage']
for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature])
    
    
encoded_info = {}
for feature in categorical_features:
    encoder = label_encoders[feature]
    classes = list(encoder.classes_)
    encoded_values = list(encoder.transform(classes))
    encoded_info[feature] = dict(zip(classes, encoded_values))

# Print encoded information for each categorical feature
for feature in categorical_features:
    print(f"Feature: {feature}")
    print(f"Encode Values with Classes: {encoded_info[feature]}")
    print()

copy = df.drop(['N_Days', 'Status', 'Drug', 'Stage'], axis=1)  # Dropping unnecessary columns
first_row = copy.iloc[[2]]
print(first_row)
pred = loaded_booster.predict(xgb.DMatrix(first_row))

print(pred)
input_features = ["Age","Sex","Ascites","Hepatomegaly","Spiders","Edema",
                  "Bilirubin","Cholesterol","Albumin","Copper","Alk_Phos","SGOT",
                  "Tryglicerides","Platelets","Prothrombin"]

def preprocess_data(input_data):
    try:
        input_data['Sex'] = input_data['Sex'].map({'F': 0, 'M': 1})
        input_data['Ascites'] = input_data['Ascites'].map({'N': 0, 'Y': 1})
        input_data['Hepatomegaly'] = input_data['Hepatomegaly'].map({'N': 0, 'Y': 1})
        input_data['Spiders'] = input_data['Spiders'].map({'N': 0, 'Y': 1})
        input_data['Edema'] = input_data['Edema'].map({'N': 0, 'S': 1, 'Y': 2})
        return input_data

    except Exception as e:
        raise ValueError(f"Error preprocessing input data: {str(e)}")






@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400
    
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data], columns=input_features)
        input_data = preprocess_data(input_data)
        dmatrix = xgb.DMatrix(input_data)
        prediction = loaded_booster.predict(dmatrix)
        predicted_class = int(prediction[0]) 
        print(predicted_class)  
        print("Predicted ")
        return jsonify({'predicted_class': predicted_class})
    except ValueError as ve: 
        return jsonify({'error': str(ve)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)