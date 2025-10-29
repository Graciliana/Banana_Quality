import pandas as pd
import os
import zipfile
import streamlit as st
import sys


# Adiciona o caminho do diretório src no Python Path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def carregar_dataset_zip(
    zip_path, nome_csv="banana_quality.csv", encoding="utf-8", sep=","
):
    """
    Extrai e carrega um arquivo CSV de dentro de um ZIP.

    Parâmetros:
        zip_path (str): Caminho do arquivo ZIP.
        nome_csv (str): Nome do arquivo CSV dentro do ZIP.
        encoding (str): Codificação do CSV.
        sep (str): Separador do CSV.

    Retorna:
        pd.DataFrame: DataFrame carregado, ou None se houver erro.
    """
    if not os.path.exists(zip_path):
        st.error(f"❌ Arquivo ZIP não encontrado: {zip_path}")
        return None

    extract_path = os.path.join(os.path.dirname(zip_path), "extracted")
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
        #st.info(f"📦 Arquivos extraídos para: `{extract_path}`")

    csv_path = os.path.join(extract_path, nome_csv)
    if not os.path.exists(csv_path):
        st.error(f"❌ CSV '{nome_csv}' não encontrado em {extract_path}")
        return None

    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
    #st.success("✅ Dataset carregado com sucesso!")
    return df

def renomear_colunas(df, colunas_dict=None):
    """
    Renomear as colunas de um DataFrame.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        colunas_dict (dict, opcional): Dicionário {coluna_antiga: coluna_nova}.
                                       Se None, usa um mapeamento padrão.
                                        realizar a tradução para o portugues das colunas

    Retorna:
        pd.DataFrame: DataFrame com colunas renomeadas.

    """
    if colunas_dict is None:
        colunas_dict = {
            "Size": "Tamanho",
            "Weight": "Peso",
            "Sweetness": "Doçura",
            "Softness": "Maciez",
            "HarvestTime": "Época de Colheita",
            "Ripeness": "Maturidade",
            "Acidity": "Acidez",
            "Quality": "Qualidade",
        }
    df = df.rename(columns=colunas_dict)
    return df

def mapear_qualidade(df, coluna="Qualidade"):
    """
    Mapeia os valores da coluna de qualidade de inglês para português.

    Parâmetros:
        df (pd.DataFrame): DataFrame original.
        coluna (str): Nome da coluna a ser mapeada.

    Retorna:
        pd.DataFrame: DataFrame com os valores mapeados.
    """
    mapeamento = {"Good": "Bom", "Bad": "Ruim"}
    df[coluna] = df[coluna].map(mapeamento)
    return df
