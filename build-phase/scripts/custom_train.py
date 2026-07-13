import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    args = parser.parse_args()

    # load training data
    train_files = [os.path.join(args.train, f) for f in os.listdir(args.train) if f.endswith('.csv')]
    train_data = pd.concat([pd.read_csv(f, header=None) for f in train_files])

    # first column is target
    y_train = train_data.iloc[:, 0]
    X_train = train_data.iloc[:, 1:]

    # train
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # evaluate on training data
    preds = clf.predict(X_train)
    acc = accuracy_score(y_train, preds)
    print(f"Training accuracy: {acc:.4f}")

    # save model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
