import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

def run_preprocessing():
    if os.path.exists("heart_raw/heart_raw.csv"):
        input_path = "heart_raw/heart_raw.csv"
        output_path = "preprocessing/heart_preprocessed.csv"
    elif os.path.exists("../heart_raw/heart_raw.csv"):
        input_path = "../heart_raw/heart_raw.csv"
        output_path = "heart_preprocessed.csv"
    else:
        print("ERROR: Dataset heart_raw.csv tidak ditemukan!")
        sys.exit(1)

    # Load Data
    df = pd.read_csv(input_path)
    
    # Preprocessing
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['num'], inplace=True)
    
    scaler = StandardScaler()
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Preprocessing Berhasil! Disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
