import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# ============== Load Data ==============
df = pd.read_csv("student-performance.csv")

# ============== EDA ==============
print(df.describe())

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Korelasi antar fitur")
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['math score'], kde=True, bins=20, color='teal')
plt.title("Distribusi nilai Matematika")
plt.xlabel("Skor")
plt.ylabel("Frekuensi")
plt.show()

# ============== Regresi ==============
X_reg = df[['reading score', 'writing score']]
y_reg = df['math score']
model_reg = LinearRegression().fit(X_reg, y_reg)

plt.figure(figsize=(10,6))
sns.scatterplot(x=X_reg['reading score'], y=y_reg, color='blue')
plt.plot(X_reg['reading score'], model_reg.predict(X_reg), color='red')
plt.title("Prediksi nilai Matematika")
plt.xlabel("Reading Score")
plt.ylabel("Math Score")
plt.show()

# ============== Clustering ==============
X_clust = df[['math score', 'reading score', 'writing score']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_clust)

plt.figure(figsize=(10,6))
sns.scatterplot(x='math score', y='reading score', hue='Cluster', palette='Set2', data=df)
plt.title("Clustering siswa")
plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.show()

# ============== Klasifikasi ==============
df['Target'] = (df['math score'] > 70).astype(int)
X_clf = df[['reading score', 'writing score']]
y_clf = df['Target']

model_clf = RandomForestClassifier(random_state=42)
model_clf.fit(X_clf, y_clf)
importances = pd.Series(model_clf.feature_importances_, index=X_clf.columns)

plt.figure(figsize=(10,6))
importances.plot(kind='barh', color='orange')
plt.title("Feature Importance")
plt.xlabel("Skor")
plt.show()