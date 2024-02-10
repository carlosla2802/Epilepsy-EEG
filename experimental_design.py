import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from dataset_preparation import EpilepticDataset  # Asegúrate de que este sea el nombre correcto de tu archivo y clase
import numpy as np
import matplotlib.pyplot as plt
import json

from models import LSTMBasedFusion, HybridLSTMCNN, TrainableChannelFusion


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
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.tolist())
            all_preds.extend(predicted.tolist())

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Manejar el caso donde solo hay una clase en y_true
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc_auc = float('nan')  # o cualquier otro valor que prefieras    
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix.tolist()
    }


def plot_metrics(metrics_history, model_name, split_type, num_epochs, k):
    epochs = range(1, num_epochs + 1)

    # Calcular el promedio de las métricas por época a lo largo de todos los folds
    avg_train_loss = [np.mean(metrics_history['train_loss'][i::k]) for i in range(num_epochs)]
    avg_val_f1 = [np.mean(metrics_history['val_f1'][i::k]) for i in range(num_epochs)]

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_loss, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Average Training Loss Over Epochs - {model_name} - {split_type}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_val_f1, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'Average Validation F1 Score Over Epochs - {model_name} - {split_type}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_{split_type}_learning_curve.png')
    plt.close()


def k_fold_cross_validation(dataset, k, model_class, num_epochs, split_type, device):
    metrics_history = {'train_loss': [], 'val_f1': []}
    val_metrics = []

    if split_type == "window":
        kf = KFold(n_splits=k, shuffle=True)
        splits = kf.split(dataset)

    elif split_type == "seizure":
        group_kfold = GroupKFold(n_splits=k)
        groups = dataset.combined_data['filename_interval'].values  # Asumiendo que este campo identifica los episodios de convulsión
        splits = group_kfold.split(dataset, groups=groups)

    elif split_type == "patient":
        group_kfold = GroupKFold(n_splits=k)
        groups = dataset.combined_data['patient_id'].values
        splits = group_kfold.split(dataset, groups=groups)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"Training {model_class.__name__} - Split: {split_type} - Fold: {fold} of {k}")
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=64, shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=64, shuffle=False)

        model = model_class(input_size=128, hidden_size=128, num_classes=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        fold_val_metrics = []
        for epoch in range(num_epochs):
            print(f"Fold {fold} - Epoch {epoch + 1} of {num_epochs}")
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            eval_metrics = evaluate_model(model, test_loader, device)
            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_f1'].append(eval_metrics['f1_score'])
            fold_val_metrics.append(eval_metrics)
        
        val_metrics.append(fold_val_metrics)

    plot_metrics(metrics_history, model_class.__name__, split_type, num_epochs, k)

    # Calcular promedios y desviaciones estándar
    avg_val_metrics = {}
    std_val_metrics = {}
    for key in fold_val_metrics[0].keys():
        fold_metric_values = [metrics[key] for fold_metrics in val_metrics for metrics in fold_metrics]
        avg_val_metrics[key] = np.mean(fold_metric_values)
        std_val_metrics[key] = np.std(fold_metric_values)

    return avg_val_metrics, std_val_metrics


def evaluate_on_test_set(model, test_dataset, device):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return evaluate_model(model, test_loader, device)

def train_and_evaluate_all_models(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [LSTMBasedFusion, HybridLSTMCNN ,TrainableChannelFusion]
    splits = ["patient", "window", "seizure"]
    k = 4  # Define el número de folds
    num_epochs = 5 #Define el número de epochs

    # Crear y dividir el dataset
    full_dataset = EpilepticDataset(data_dir + "/MetaData", data_dir)
    train_val_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size])
    
    for model_class in models:
        for split_type in splits:
            key = f"{model_class.__name__}_{split_type}"
            print(f"\nTraining and evaluating {key}")

            # Validación cruzada en el conjunto de entrenamiento y validación
            avg_val_metrics, std_val_metrics = k_fold_cross_validation(train_val_dataset, k, model_class, num_epochs, split_type, device)

            # Reentrenar final_model con todo el conjunto de entrenamiento y validación
            final_model = model_class(input_size=128, hidden_size=128, num_classes=2).to(device)
            full_train_loader = DataLoader(train_val_dataset, batch_size=64, shuffle=True)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
            for epoch in range(num_epochs):
                train_model(final_model, full_train_loader, criterion, optimizer, device)

            # Evaluación final en el conjunto de pruebas independiente
            test_metrics = evaluate_on_test_set(final_model, test_dataset, device)
    
            # Guardar resultados de validación y test en un solo archivo JSON
            all_results = {
                'validation_metrics': {
                    'average': avg_val_metrics,
                    'standard_deviation': std_val_metrics
                },
                'test_metrics': test_metrics
            }
            with open(f'{key}_results.json', 'w') as f:
                json.dump(all_results, f)

    print("Todos los modelos y splits han sido evaluados y guardados.")



# Ejemplo de uso y guardado de resultados
data_dir = "annotated_windows"
all_results = train_and_evaluate_all_models(data_dir)