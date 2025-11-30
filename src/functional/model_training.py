import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(model_name):
    data_path = 'src/data/landmarks.csv'
    label_path = 'src/data/landmark_labels.csv'

    data_df = pd.read_csv(data_path)
    label_df = pd.read_csv(label_path)
    y = label_df.values[:,0]

    x_train,x_test,y_train,y_test = train_test_split(data_df.values, y, test_size=0.2, shuffle= True, stratify=y)

    if model_name == 'RandomForest':
        model = RandomForestClassifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    score = accuracy_score(y_pred, y_test)

    joblib.dump(model,f"src/functional/{model_name}.pkl")

    return score

if __name__=="__main__":
    print(train_model("RandomForest"))


