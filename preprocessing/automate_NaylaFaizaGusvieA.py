"""
Konversi otomatis dari notebook Eksperimen_NaylaFaizaGusvieA.ipynb
Menjalankan seluruh tahap preprocessing secara otomatis.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os, warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    # Bagian 3 — Memuat Dataset
    df = pd.read_csv(filepath)
    print(f'[LOAD] Dataset dimuat: {df.shape[0]:,} baris, {df.shape[1]} kolom')
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Bagian 5 — Data Preprocessing
    df_p = df.copy()
    num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    cat_features = ['gender', 'smoking_history']

    # 5.2 Missing Values
    df_p[num_features] = SimpleImputer(strategy='median').fit_transform(df_p[num_features])
    df_p[cat_features] = SimpleImputer(strategy='most_frequent').fit_transform(df_p[cat_features])
    print('[PREP] Missing values ditangani.')

    # 5.3 Duplikat
    before = len(df_p)
    df_p.drop_duplicates(inplace=True)
    print(f'[PREP] Duplikat dihapus: {before - len(df_p)} baris')

    # 5.4 Encoding
    df_p['gender']          = LabelEncoder().fit_transform(df_p['gender'])
    df_p['smoking_history'] = LabelEncoder().fit_transform(df_p['smoking_history'])
    print('[PREP] Encoding kategorikal selesai.')

    # 5.5 Outlier (IQR Clipping)
    for col in num_features:
        Q1, Q3 = df_p[col].quantile(0.25), df_p[col].quantile(0.75)
        IQR = Q3 - Q1
        df_p[col] = df_p[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    print('[PREP] Outlier di-clip dengan metode IQR.')

    # 5.6 Standarisasi
    df_p[num_features] = StandardScaler().fit_transform(df_p[num_features])
    print('[PREP] Standarisasi fitur numerik selesai.')

    return df_p


def split_and_save(df_p: pd.DataFrame, out: str = 'diabetes_preprocessing') -> tuple:
    X = df_p.drop('diabetes', axis=1)
    y = df_p['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(out, exist_ok=True)
    X_train.to_csv(f'{out}/X_train.csv', index=False)
    X_test.to_csv( f'{out}/X_test.csv',  index=False)
    y_train.to_csv(f'{out}/y_train.csv', index=False)
    y_test.to_csv( f'{out}/y_test.csv',  index=False)
    print(f'[SAVE] Data disimpan ke {out}/')
    print(f'       Train: {X_train.shape} | Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else '../diabetes_prediction_dataset.csv'
    print('=== AUTOMATE PREPROCESSING — DIABETES DATASET ===')
    df   = load_data(fp)
    df_p = preprocess(df)
    split_and_save(df_p)
    print('=== PREPROCESSING SELESAI! ===')
