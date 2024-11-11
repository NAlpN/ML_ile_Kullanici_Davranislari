import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('user_behavior_dataset.csv')

X_test = df[['Age', 'Gender']]
y_true = df['Operating System']

X_test['Gender'] = X_test['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_true, y_pred)
print(f"Model Doğruluk Oranı: {accuracy:.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.savefig('karisiklikMatrisi.png')