import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud

# Cargar el dataset
dataset = pd.read_excel("Dataset_SaludMental.csv")

# Limpiar nombres de columnas
dataset.columns = dataset.columns.str.strip()

# Separar columnas numéricas
numeric_cols = [
    "¿Cuántas horas duermes en promedio cada noche?",
    "En una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés durante el último mes?",
    "¿Cuántas horas al día dedicas al estudio fuera del horario de clases?",
]
num_data = dataset[numeric_cols]

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(num_data.describe())

# Visualización inicial
for col in numeric_cols:
    plt.figure()
    sns.histplot(num_data[col], kde=True)
    plt.title(f"Distribución de {col}")
    plt.show()

# Nube de palabras
text_data = dataset.drop(columns=numeric_cols + ["Marca temporal"])
all_text = " ".join(text_data.apply(lambda x: " ".join(x.astype(str)), axis=1))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Nube de palabras")
plt.show()

# Normalización y K-means clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_data)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
num_data["Cluster"] = clusters

plt.figure()
sns.histplot(data=num_data, x="¿Cuántas horas duermes en promedio cada noche?", hue="Cluster", multiple="stack")
plt.title("Distribución de horas de sueño por clúster")
plt.show()

# Clasificacion
num_data["Estrés Alto"] = num_data[
    "En una escala del 1 al 10, ¿cómo calificarías tu nivel de estrés durante el último mes?"
].apply(lambda x: 1 if x >= 7 else 0)

X = num_data.drop(["Cluster", "Estrés Alto"], axis=1)
y = num_data["Estrés Alto"]

X_noisy = X + np.random.normal(0, 0.5, X.shape)
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)


# Reporte de clasificación
print("Resultados de la clasificación")
print(classification_report(y_test, y_pred))
