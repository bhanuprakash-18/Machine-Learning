
---

### 🔹 `knn/knn.ipynb`

A notebook with explanations and visualizations. Here's a [simplified preview](https://gist.github.com) of what it should include (you can convert `knn.py` to `.ipynb` using Jupyter or VS Code).

---

### 🔹 `knn/knn.py`

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.preprocessing import preprocess_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Load and preprocess data
X, y = preprocess_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save metrics
with open("results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("images/knn_confusion_matrix.png")
plt.close()

