import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load




HOUSE_DATA = {
    6:  {"occupancy": 2, "construction_year": 2005,   "appliances": 49, "house_type": 0, "bedrooms": 4},
    15: {"occupancy": 1, "construction_year": 1970,   "appliances": 19, "house_type": 1, "bedrooms": 3},
    18: {"occupancy": 2, "construction_year": 1970,   "appliances": 34, "house_type": 0, "bedrooms": 3},
    19: {"occupancy": 4, "construction_year": 1955,   "appliances": 26, "house_type": 1, "bedrooms": 3},
    20: {"occupancy": 2, "construction_year": 1970,   "appliances": 39, "house_type": 0, "bedrooms": 3},
}

pred = [3 , 1990, 30, 1, 3]

df = pd.DataFrame.from_dict(HOUSE_DATA, orient='index')
df.index.name = 'house_id'
Y = df.index.to_numpy()   
X = df.to_numpy()        

print(df.head())
print(X)
print(Y)

classifier = KNeighborsClassifier(n_neighbors=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
classifier.fit(X_scaled, Y)
classifier.fit(X_scaled, Y)

pred_scaled = scaler.transform([pred])
prediction = classifier.predict(pred_scaled)
print(prediction)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pred_pca = pca.transform(pred_scaled)

plt.figure(figsize=(7, 5))
for i, house_id in enumerate(Y):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], s=100)
    plt.annotate(f"House {house_id}", (X_pca[i, 0], X_pca[i, 1]), textcoords="offset points", xytext=(5, 5))

plt.scatter(pred_pca[0, 0], pred_pca[0, 1], s=150, marker='*', color='black', label=f"New House {prediction[0]}")
plt.legend()
plt.title("KNN - PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

#dump(classifier,   "C:\\Users\\nolan\\OneDrive\\Documents\\GitHub\\PE2-AI\\HOUSEHOLDPIPELINE\\Top7_XGBoost_Active\\models\\knn_classifier.joblib")
#dump(scaler,     r"C:\Users\nolan\OneDrive\Documents\GitHub\PE2-AI\HOUSEHOLDPIPELINE\Top7_XGBoost_Active\models\knn_scaler.joblib")