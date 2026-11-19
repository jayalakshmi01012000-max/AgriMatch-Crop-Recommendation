"""Train a crop recommendation model. Run: python train.py"""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

df = pd.read_csv("Crop_recommendation.csv")
X, y = df[FEATURES], df["label"]

# train/test split: 80% train, 20% held out
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
best_name, best_model, best_acc = None, None, 0
for name, model in candidates.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"{name}: validation accuracy = {acc:.4f}")
    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print(f"\nBest: {best_name}")
print(classification_report(y_val, best_model.predict(X_val)))

# refit the winner on all data and save
best_model.fit(X, y)
joblib.dump(best_model, "model.joblib")
print("Saved model.joblib")
