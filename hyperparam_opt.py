import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_slice, plot_optimization_history
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import numpy as np

# Importa tus archivos y modelos personalizados
from dataset_preparation import EpilepticDataset
from models import TrainableChannelFusion2

# Configura tu dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.tolist())
            all_preds.extend(predicted.tolist())
    
    # Calcula la sensibilidad
    sensitivity = recall_score(all_labels, all_preds)
    return sensitivity

def objective(trial):
    # Espacio de búsqueda de hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    input_size = 128  # Esto puede variar según tus datos
    num_classes = 2  # Clasificación binaria
    k = 4  # Número de folds en la validación cruzada
    num_epochs = 5  # Número de epochs para el entrenamiento

    # Crear dataset y realizar validación cruzada
    data_dir = "annotated_windows"
    dataset = EpilepticDataset(data_dir + "/MetaData", data_dir)  # Asegúrate de cargar tu dataset correctamente
    
    group_kfold = GroupKFold(n_splits=k)
    groups = dataset.combined_data['patient_id'].values
    splits = group_kfold.split(dataset, groups=groups)

    sensitivities  = []

    for train_index, test_index in splits:
        train_loader = DataLoader(Subset(dataset, train_index), batch_size=64, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_index), batch_size=64, shuffle=False)

        # Crear y configurar el modelo
        model = TrainableChannelFusion2(input_size, hidden_size, num_classes, num_layers, bidirectional, dropout_rate).to(device)
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        # Entrenamiento y evaluación
        for epoch in range(num_epochs):
            train_model(model, train_loader, criterion, optimizer, device)
        sensitivity  = evaluate_model(model, test_loader, device)
        sensitivities.append(sensitivity)

    # Devolver la precisión media
    return np.mean(sensitivities)


# Crear un estudio de Optuna y optimizar
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, show_progress_bar=True)  # Ejecutar 10 pruebas

# Diagrama de Coordenadas Paralelas
fig = plot_parallel_coordinate(study)
fig.write_image("parallel_coordinate_plot.png")

# Gráfico de Contorno
fig = plot_contour(study)
fig.write_image("contour_plot.png")

# Gráfico de Caja (Slice Plot)
fig = plot_slice(study)
fig.write_image("slice_plot.png")

# Historial de Optimización
fig = plot_optimization_history(study)
fig.write_image("optimization_history.png")


# Mejores hiperparámetros encontrados
best_params = study.best_params
print("Mejores hiperparámetros:", best_params)
