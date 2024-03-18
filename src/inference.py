from joblib import load
import pandas as pd

def get_prediction(**kwargs):
    clf = load('models/model.joblib')
    features = load('models/raw_features.joblib')
    pred_df = pd.DataFrame(kwargs, index=[0])
    pred = clf.predict(pred_df[features])
    return pred[0]
