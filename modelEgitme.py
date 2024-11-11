import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('user_behavior_dataset.csv')

X = df[['Age', 'Gender']]
y = df['Operating System']

X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB()
}

accuracies = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    print(f"{model_name} Doğruluk Oranı: {accuracy:.4f}")

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"\nEn iyi model: {best_model_name} ({accuracies[best_model_name]:.4f} doğruluk oranı ile)")

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)


plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
plt.title("Modellerin Doğruluk Oranlarının Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Doğruluk Oranı")
plt.xticks(rotation=45)
plt.savefig('modelKarsilastirmasi.png')