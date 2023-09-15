# Importamos las librerías necesarias
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import seaborn as sns
sns.set_style("darkgrid")

# Cargamos el dataset de "breast_cancer_detection" usando sklearn
data = load_breast_cancer()

# Extraemos los Features "X" y la variable objetivo "Y"
X = data.data
Y = data.target

# Usamos sklearn para dividir el set en test, train y validation
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#-------------------------------------------------------------------------------------------------------------
#   Modelo 1: usando los hiperparámetros predeterminados
#-------------------------------------------------------------------------------------------------------------

# Creamos el modelo1 con los hiperparámetros predeterminados
model1 = SVC()
model1.fit(X_train, y_train)
predictions_val = model1.predict(X_validation)

# Calculamos la matriz de confusión
confusion = confusion_matrix(y_validation, predictions_val)

# Creamos la gráfica de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[-1, 1])

# Calculamos el reporte de clasificación 
class_report1 = classification_report(y_validation, predictions_val)
print("Classification Report for Default Model:\n", class_report1)

# Calculamos la curva de aprendizaje para el modelo con los hiperparámetros predeterminados. 
train_sizes1, train_scores1, val_scores1 = learning_curve(model1, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 20), cv=5)

# Calculamos la media y la desviación estándar para cada entrenamiento
train_mean1 = np.mean(train_scores1, axis=1)
train_std1 = np.std(train_scores1, axis=1)
val_mean1 = np.mean(val_scores1, axis=1)
val_std1 = np.std(val_scores1, axis=1)

#-------------------------------------------------------------------------------------------------------------
#   Modelo 2: Using improved hyperparameters
#-------------------------------------------------------------------------------------------------------------

# Creamos un nuevo modelo con un valor distinto de "C" y un kernel lineal 
model2 = SVC(C=0.1, kernel="linear")
model2.fit(X_train, y_train)
predictions_val2 = model2.predict(X_validation)

# Calculamos la matriz de confusión
confusion2 = confusion_matrix(y_validation, predictions_val2)

# Creamos la gráfica de la matriz de confusión
disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion2, display_labels=[-1, 1])

# Calculamos el reporte de clasificación 
class_report2 = classification_report(y_validation, predictions_val2)
print("Classification Report for Improved Model:\n", class_report2)

# Calculamos la curva de aprendizaje para el modelo con los hiperparámetros mejorados.
train_sizes2, train_scores2, val_scores2 = learning_curve(model2, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 20), cv=5)

# Calculamos la media y la desviación estándar para cada entrenamiento
train_mean2 = np.mean(train_scores2, axis=1)
train_std2 = np.std(train_scores2, axis=1)
val_mean2 = np.mean(val_scores2, axis=1)
val_std2 = np.std(val_scores2, axis=1)

# Creamos subplots para poner todas las gráficas
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Ponemos el plot de la matriz de confusión del modelo con los parámetros predeterminados
disp.plot(cmap='Blues', values_format='d', ax=axs[0, 0])
axs[0, 0].set_title('Confusion Matrix for Default Model')

# Ponemos el reporte de clasificación en el plot
axs[0, 1].text(0.5, 0.5, class_report1, fontsize=10, ha='left', va='center')
axs[0, 1].axis('off')
axs[0, 1].set_title('Classification Metrics for Default Model')

# Ponemos el plot de la curva de aprendizaje
axs[0, 2].plot(train_sizes1, train_mean1, label='Training Score', color='blue', marker='o')
axs[0, 2].fill_between(train_sizes1, train_mean1 - train_std1, train_mean1 + train_std1, color='blue', alpha=0.25)
axs[0, 2].plot(train_sizes1, val_mean1, label='Validation Score', color='red', marker='o')
axs[0, 2].fill_between(train_sizes1, val_mean1 - val_std1, val_mean1 + val_std1, color='red', alpha=0.25)
axs[0, 2].set_xlabel('Number of Training Examples')
axs[0, 2].set_ylabel('Accuracy (or another metric)')
axs[0, 2].set_title('Learning Curve for Default Model')
axs[0, 2].legend(loc='best')
axs[0, 2].grid(True)

# Ponemos el plot de la matriz de confusión del modelo con los parámetros mejorados
disp2.plot(cmap='Blues', values_format='d', ax=axs[1, 0])
axs[1, 0].set_title('Confusion Matrix for Improved Model')

# Ponemos el reporte de clasificación en el plot
axs[1, 1].text(0.5, 0.5, class_report2, fontsize=10, ha='left', va='center')
axs[1, 1].axis('off')
axs[1, 1].set_title('Classification Metrics for Improved Model')

# Ponemos el plot de la curva de aprendizaje
axs[1, 2].plot(train_sizes2, train_mean2, label='Training Score', color='blue', marker='o')
axs[1, 2].fill_between(train_sizes2, train_mean2 - train_std2, train_mean2 + train_std2, color='blue', alpha=0.25)
axs[1, 2].plot(train_sizes2, val_mean2, label='Validation Score', color='red', marker='o')
axs[1, 2].fill_between(train_sizes2, val_mean2 - val_std2, val_mean2 + val_std2, color='red', alpha=0.25)
axs[1, 2].set_xlabel('Number of Training Examples')
axs[1, 2].set_ylabel('Accuracy (or another metric)')
axs[1, 2].set_title('Learning Curve for Improved Model')
axs[1, 2].legend(loc='best')
axs[1, 2].grid(True)

# Creammos la gráfica
plt.tight_layout()
plt.show()
