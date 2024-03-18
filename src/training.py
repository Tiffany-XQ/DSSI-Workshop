import argparse
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier

from joblib import dump, load

import data_processor

logging.basicConfig(format='%(asctime)s || %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO)

features = ['gender','age','pclass','embarked','fare','sibsp','parch']
numeric_features = ['age','fare','sibsp','parch']
categorical_features = ['gender','pclass','embarked']
label = 'survived'

def run(data_path, model_path, f1_criteria):

    # Pre-processing
    logging.info('Process Data...')
    df = data_processor.run(data_path)
    
    numeric_transformer = MinMaxScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    # Train-Test Split
    logging.info('Start Train-Test Split...')
    X_train, X_test, y_train, y_test = train_test_split(df[features], 
                                                        df[label], 
                                                        test_size=0.2, 
                                                        random_state=2024)
    
    # Train Classifier
    logging.info('Start Training...')
    gradient_boost = GradientBoostingClassifier(n_estimators=100,
                                                random_state=2024)
    
    clf = Pipeline(steps=[("preprocessor", preprocessor),\
                          ("binary_classifier", gradient_boost)
                         ])
    clf.fit(X_train, y_train)
    
    #Evaluate and Deploy
    logging.info('Evaluate...')
    score = f1_score(y_test, clf.predict(X_test), average='weighted')
    if score >= f1_criteria:
        logging.info('Deploy...')
        dump(clf, model_path+'model.joblib')
        dump(features, model_path+'raw_features.joblib')
    
    logging.info('Training completed.')
    return None

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--f1_criteria", type=float)
    args = argparser.parse_args()
    run(args.data_path, args.model_path, args.f1_criteria)