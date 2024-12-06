import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_and_process_data():
    """Carga y procesa el dataset inicial"""
    # Cargar el dataset original
    filepath = os.path.join(BASE_DIR, 'dataset', 'Base_clientes_Monopoly.xlsx')
    ds = pd.read_excel(filepath)
    
    # Procesar el dataset
    ds.columns = ds.iloc[0]
    ds = ds[1:]
    ds.reset_index(drop=True, inplace=True)
    
    return ds

def prepare_data(ds, sample_size=None):
    """Prepara los datos para el entrenamiento"""
    # Definir las características que vamos a usar
    features = ['Renta', 'Edad', 'Antiguedad', 'Region', 'Sexo']
    target = 'target'
    
    # Verificar si las columnas existen
    missing_columns = [col for col in features + [target] if col not in ds.columns]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en el archivo: {missing_columns}")
    
    # Convertir columnas numéricas
    ds['Renta'] = pd.to_numeric(ds['Renta'], errors='coerce')
    ds['Edad'] = pd.to_numeric(ds['Edad'], errors='coerce')
    ds['Antiguedad'] = pd.to_numeric(ds['Antiguedad'], errors='coerce')
    
    # Manejar valores NaN
    ds = ds.dropna(subset=features + [target])
    
    # Crear variables dummy
    X = pd.get_dummies(ds[features], drop_first=True)
    y = ds[target].astype(int)

    # Tomar una muestra si se especifica
    if sample_size and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model():
    """Entrena y guarda el modelo"""
    # Cargar y procesar el dataset
    print("Cargando datos...")
    ds = load_and_process_data()
    
    # Preparar los datos
    print("Preparando datos...")
    X_train, X_test, y_train, y_test = prepare_data(ds, sample_size=None)  # Usar todos los datos
    
    # Entrenar el modelo
    print("Entrenando modelo...")
    modelo = DecisionTreeClassifier(
        max_depth=10,  # Profundidad máxima del árbol
        min_samples_split=5,  # Mínimo de muestras para dividir un nodo
        min_samples_leaf=5,  # Mínimo de muestras en hojas
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    train_score = modelo.score(X_train, y_train)
    test_score = modelo.score(X_test, y_test)
    print(f"\nPrecisión en entrenamiento: {train_score:.2f}")
    print(f"Precisión en prueba: {test_score:.2f}")
    
    # Guardar el modelo
    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    modelo_path = os.path.join(checkpoint_dir, 'modelo.pkl')
    joblib.dump(modelo, modelo_path)
    print(f"\nModelo guardado en: {modelo_path}")
    
    # Guardar las columnas esperadas
    columnas_path = os.path.join(checkpoint_dir, 'columnas.pkl')
    joblib.dump(X_train.columns.tolist(), columnas_path)
    print(f"Columnas guardadas en: {columnas_path}")
    
    return modelo

def load_model():
    """Carga el modelo guardado"""
    modelo_path = os.path.join(BASE_DIR, 'checkpoints', 'modelo.pkl')
    if not os.path.exists(modelo_path):
        raise FileNotFoundError("El modelo no se encuentra. Ejecuta train_model primero.")
    return joblib.load(modelo_path)

if __name__ == "__main__":
    try:
        print("Iniciando entrenamiento del modelo...")
        modelo = train_model()
        print("\nProceso completado exitosamente!")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")