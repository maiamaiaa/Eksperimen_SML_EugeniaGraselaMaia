import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    if os.path.exists("heart_raw/heart_raw.csv"):
        input_path = "heart_raw/heart_raw.csv"
        output_path = "preprocessing/heart_preprocessed.csv"
    else:
        input_path = "../heart_raw/heart_raw.csv"
        output_path = "heart_preprocessed.csv"
    
    print(f"Memproses data dari: {input_path}")
    df = pd.read_csv(input_path)
    
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=['num'], inplace=True)
    
    scaler = StandardScaler()
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    df.to_csv(output_path, index=False)
    print(f"Berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
