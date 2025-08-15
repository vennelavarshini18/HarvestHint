import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Crop_recommendation.csv")

data = pd.read_csv(csv_path)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)  

model=RandomForestClassifier()

model.fit(X_train,y_train)

predictions=model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as crop_model.pkl")
