import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel, RFECV
import joblib
import json


class ModelDevelop:
    
    """
    Clase para desarrollar modelos de machine learning, incluyendo balanceo de datos,
    selección de características, optimización de hiperparámetros y evaluación de desempeño.
    """

    def __init__(self):
        pass

    def balance_data(self, X, y): 
        """
        Balancea los datos aplicando primero sobremuestreo con SMOTE y luego submuestreo.
        
        Args:
            X (pd.DataFrame): Conjunto de características.
            y (pd.Series): Etiquetas de clase.
        
        Returns:
            tuple: Conjunto de características y etiquetas balanceadas.
        """
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        under_sampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = under_sampler.fit_resample(X_smote, y_smote)
        return X_balanced, y_balanced

    def train_test_split_df(self, X_train, y_train):
        """
        Divide el conjunto de entrenamiento en subconjuntos de entrenamiento interno y validación.
        
        Args:
            X_train (pd.DataFrame): Conjunto de características de entrenamiento.
            y_train (pd.Series): Etiquetas de entrenamiento.
        
        Returns:
            tuple: Datos de entrenamiento interno y validación.
        """
        X_train_internal, X_val, y_train_internal, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        return X_train_internal, X_val, y_train_internal, y_val
    
    def select_features_with_model(self, X, y):
        """
        Selecciona características utilizando un modelo RandomForestClassifier.
        
        Args:
            X (pd.DataFrame): Conjunto de características.
            y (pd.Series): Etiquetas de clase.
        
        Returns:
            tuple: Conjunto de características seleccionadas y nombres de las características seleccionadas.
        """
        model_temp = RandomForestClassifier(random_state=42)
        model_temp.fit(X, y)
        selector = SelectFromModel(model_temp, prefit=True, threshold="mean")
        selected_features = X.columns[selector.get_support()]
        X_selected = selector.transform(X)
        return X_selected, selected_features
    
    def grid_search_rf(self, X, y):
        """
        Realiza una búsqueda en malla para optimizar hiperparámetros de RandomForestClassifier.
        
        Args:
            X (pd.DataFrame): Conjunto de características.
            y (pd.Series): Etiquetas de clase.
        
        Returns:
            RandomForestClassifier: Modelo con los mejores hiperparámetros encontrados.
        """
        print("\n[Optimized Search - Random Forest]")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=2,  # Reducir número de particiones
            scoring='accuracy',
            n_jobs=-1  # Paralelizar
        )
        grid_search.fit(X, y)
        print("\n[Random Forest] Mejores parámetros:", grid_search.best_params_)
        return grid_search.best_estimator_
    
    def grid_search_rf_sampled(self, X, y):
        print("\n[Optimized Search - Random Forest]")
        param_grid = {
            'n_estimators': [50],  # Reduce el número de estimadores
            'max_depth': [10],  # Usa valores más simples
            'min_samples_split': [10],  # Un solo valor
            'min_samples_leaf': [1],  # Un solo valor
            'class_weight': ['balanced']
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=2,  # Mantén un número bajo de divisiones
            scoring='accuracy',
            n_jobs=-1  # Paraleliza
        )
        # Usar solo una muestra de los datos
        X_sampled = X.sample(frac=0.3, random_state=42)  # Usa 30% de los datos
        y_sampled = y.loc[X_sampled.index]
        grid_search.fit(X_sampled, y_sampled)
        print("\n[Random Forest] Mejores parámetros:", grid_search.best_params_)
        return grid_search.best_estimator_


    def train_and_evaluate_models(self, X_train, y_train, X_val, y_val, le):
        """
        Entrena y evalúa un modelo RandomForestClassifier con validación cruzada y en el conjunto de validación.
        
        Args:
            X_train (pd.DataFrame): Conjunto de características de entrenamiento.
            y_train (pd.Series): Etiquetas de entrenamiento.
            X_val (pd.DataFrame): Conjunto de características de validación.
            y_val (pd.Series): Etiquetas de validación.
            le (LabelEncoder): Codificador de etiquetas para interpretación del reporte de clasificación.
        
        Returns:
            RandomForestClassifier: Modelo entrenado.
        """
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)

        print("\n[Random Forest - Validación Cruzada]")
        #rf_model = self.grid_search_rf(X_train_balanced, y_train_balanced)
        rf_model = self.grid_search_rf_sampled(X_train_balanced, y_train_balanced)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')
        print(f"[Random Forest] Accuracy promedio CV: {np.mean(rf_cv_scores):.4f} ± {np.std(rf_cv_scores):.4f}")

        # Validación interna para Random Forest
        rf_model.fit(X_train_balanced, y_train_balanced)
        y_val_pred_rf = rf_model.predict(X_val)
        print("\n[Evaluación en Validación Interna - Random Forest]")
        print("Accuracy:", accuracy_score(y_val, y_val_pred_rf))
        print("Classification Report:\n", classification_report(y_val, y_val_pred_rf, target_names=le.classes_))

        return rf_model
    
    def predict_on_test_set(self, model, X_test):
        """
        Realiza predicciones en el conjunto de prueba utilizando el modelo entrenado.

        Args:
            model (RandomForestClassifier): Modelo previamente entrenado.
            X_test (pd.DataFrame): Conjunto de características del test set.

        Returns:
            np.ndarray: Predicciones del modelo.
        """
        try:
            logging.info("Realizando predicciones en el conjunto de prueba...")
            y_pred = model.predict(X_test)
            return y_pred
        except Exception as e:
            logging.error(f"Error durante la predicción: {e}")
            raise

    def get_model_metrics(self, model, X_val, y_val, le):
        """
        Calcula y devuelve métricas del modelo en formato JSON.
        
        Args:
            model (RandomForestClassifier): Modelo entrenado.
            X_val (pd.DataFrame): Conjunto de características de validación.
            y_val (pd.Series): Etiquetas de validación.
            le (LabelEncoder): Codificador de etiquetas.
        
        Returns:
            dict: Diccionario con métricas del modelo.
        """
        y_pred = model.predict(X_val)

        # Cálculo de métricas
        accuracy = accuracy_score(y_val, y_pred)
        class_report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
        conf_matrix = confusion_matrix(y_val, y_pred).tolist()  # Convertir matriz a lista para JSON

        # Guardar métricas en un diccionario
        metrics = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix
        }

        return metrics

    def save_model(self, model, path):
        """
        Guarda un modelo en un archivo.
        
        Args:
            model: Modelo entrenado.
            path (str): Ruta donde se guardará el modelo.
        """
        try:
            joblib.dump(model, path)
            print(f"Modelo guardado en: {path}")
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
    
    
    def feature_importance(self, model, feature_names):
        """
        Calcula la importancia de las características en el modelo.
        
        Args:
            model: Modelo entrenado con feature_importances_.
            feature_names (list): Lista de nombres de las características.
        
        Returns:
            pd.DataFrame: DataFrame con la importancia de cada característica.
        """
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return importance_df
