import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    input_path = "heart_raw/heart_raw.csv"
    output_path = "preprocessing/heart_preprocessed.csv"
    
    if not os.path.exists(input_path):
        input_path = "../heart_raw/heart_raw.csv"
        output_path = "heart_preprocessed.csv"

    # Load
    df = pd.read_csv(input_path)
    
    # 1. Handling Missing Values
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # 2. Handling Duplicates
    df.drop_duplicates(inplace=True)
    
    # 3. Binning Target (Binarisasi)
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['num'], inplace=True)
    
    # 4. Standarisasi Fitur
    scaler = StandardScaler()
    features_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # 5. Save
    df.to_csv(output_path, index=False)
    print("Otomatisasi Preprocessing Eugenia Berhasil!")

if __name__ == "__main__":
    run_preprocessing()
