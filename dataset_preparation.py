import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EpilepticDataset(Dataset):
    def __init__(self, metadata_dir, windows_dir):
        """
        Inicializa el dataset de detección de convulsiones.
        
        :param metadata_dir: Directorio que contiene los archivos parquet.
        :param windows_dir: Directorio que contiene los archivos npz.
        """
        self.metadata_dir = metadata_dir
        self.windows_dir = windows_dir
        self.combined_data = None
        self.eeg_data = {}
        self.load_data()

    def load_data(self):
        """
        Carga y combina los datos de los archivos parquet y npz.
        """

        print("Processing data and creating the Dataset:")
        parquet_files = os.listdir(self.metadata_dir)
        patient_npz_files = [f"{file_name.split('_')[0]}_seizure_EEGwindow_1.npz" for file_name in parquet_files]

        for idx, parquet_file in enumerate(parquet_files):
            print(f"Processing Parquet File {idx}")
            df = pd.read_parquet(os.path.join(self.metadata_dir, parquet_file))
            df['window_id'] = df.index
            df['patient_id'] = df['filename'].apply(lambda x: x.split("_")[0])

            self.combined_data = pd.concat([self.combined_data, df]) if self.combined_data is not None else df

        print("Finished processing Parquet files. Now processing NumPy files.")
        for idx, npz_file in enumerate(patient_npz_files):
            print(f"Processing NumPy File {idx}")
            patient_id = npz_file.split("_")[0]

            with np.load(os.path.join(self.windows_dir, npz_file), mmap_mode='r', allow_pickle=True) as data:
                self.eeg_data[patient_id] = data[data.files[0]]

    def __getitem__(self, idx):
        """
        Obtiene un ítem del dataset.
        
        :param idx: Índice del ítem.
        :return: Tupla (datos EEG, clase)
        """
        data_row = self.combined_data.iloc[idx]
        patient_id, window_id, label = data_row['patient_id'], data_row['window_id'], data_row['class']
        
        eeg_window = self.eeg_data[patient_id][window_id].astype(np.float32)
        eeg_window_tensor = torch.from_numpy(eeg_window)  # Convertir a tensor
        label_tensor = torch.tensor(label, dtype=torch.int64)  # Convertir a tensor

        return eeg_window_tensor, label_tensor

    def __len__(self):
        """
        Devuelve la longitud del dataset.
        
        :return: Longitud del dataset.
        """
        return len(self.combined_data)

#Testing
#windows_dir= "annotated_windows"
#metadata_dir = os.path.join(windows_dir, "MetaData")

#epileptic_dataset = EpilepticDataset(metadata_dir, windows_dir)
