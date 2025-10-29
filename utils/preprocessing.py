import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def tratar_valores_faltantes(df):
    """Preenche valores faltantes com média (numéricos) e moda (categóricos)."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

def normalizar_dados(df):
    """Normaliza colunas numéricas entre 0 e 1."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def codificar_categorias(df, columns=None, method='L'):
    """Codifica variáveis categóricas usando LabelEncoder (L) ou Dummies (D)."""
    encoded_df = df.copy()
    if columns is None:
        columns = encoded_df.select_dtypes(include=['object']).columns

    if method == 'L':
        encoder = LabelEncoder()
        for col in columns:
            encoded_df[col] = encoder.fit_transform(encoded_df[col])
    elif method == 'D':
        encoded_df = pd.get_dummies(encoded_df, columns=columns)
    else:
        raise ValueError("Método inválido! Escolha 'L' ou 'D'.")
    return encoded_df
