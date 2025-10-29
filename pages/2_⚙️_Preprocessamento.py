import streamlit as st
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

from utils.data_overview import data_overview
from utils.data_processing import carregar_dataset_zip, mapear_qualidade, renomear_colunas
from utils.preprocessing import normalizar_dados, tratar_valores_faltantes



# ====== Página ======
st.set_page_config(page_title="Pré-processamento", page_icon="⚙️", layout="wide")
st.title("⚙️ Pré-processamento dos Dados")
st.markdown("---")

# ====== 1️⃣ Carregar dataset ======
# =================================================================================================
# CARREGAR DATASET
# =================================================================================================
# Caminho do arquivo ZIP
zip_path = "./Data/banana_quality.zip"

st.subheader("1️⃣ Carregar Dataset Original")
df = carregar_dataset_zip(zip_path)
st.dataframe(df.head())

# ====== 2️⃣ Renomear e mapear colunas ======
st.subheader("2️⃣ Estatísticas Descritivas")
df = renomear_colunas(df)
df = mapear_qualidade(df)
df = carregar_dataset_zip(zip_path)
if df is not None:
    df = renomear_colunas(df)
    df = mapear_qualidade(df)
else:
    st.warning("⚠️ Não foi possível carregar o dataset.")
    
st.dataframe(df.describe())
#st.success("✅ Colunas renomeadas e mapeadas com sucesso!")

# ====== 3️⃣ Tratar valores faltantes ======
st.subheader("3️⃣ Tratamento de Valores Faltantes")
df = tratar_valores_faltantes(df)
st.success("✅ Valores faltantes tratados com sucesso!")

# ====== 4️⃣ Normalizar dados ======
st.subheader("4️⃣ Normalização dos Dados")
#df = normalizar_dados(df)
st.success("✅ Variáveis numéricas normalizadas com sucesso!")

# =====================================================================
# 5️⃣ Codificação das Variáveis Categóricas (LabelEncoder + Dummies)
# =====================================================================

st.subheader("5️⃣ Codificação das Variáveis Categóricas")

# Colunas a serem codificadas
COLUNAS_LABEL = ['Qualidade']  # Exemplo: variável alvo
COLUNAS_DUMMIES = [col for col in df.select_dtypes(include=['object']).columns if col not in COLUNAS_LABEL]


def codificar_variaveis_completo(data: pd.DataFrame,
                                 colunas_label: list,
                                 colunas_dummies: list) -> pd.DataFrame:
    """
    Aplica LabelEncoder e One-Hot Encoding no DataFrame.

    Parâmetros:
        data (pd.DataFrame): Dataset original
        colunas_label (list): Colunas para LabelEncoder
        colunas_dummies (list): Colunas para One-Hot Encoding

    Retorna:
        pd.DataFrame: Dataset codificado
    """
    dados_codificados = data.copy()

    # ===== Label Encoding =====
    if colunas_label:
        encoder = LabelEncoder()
        for col in colunas_label:
            dados_codificados[col] = encoder.fit_transform(dados_codificados[col])
        st.success(f"🔤 LabelEncoder aplicado em: {', '.join(colunas_label)}")

    # ===== One-Hot / Dummies =====
    if colunas_dummies:
        dummies = pd.get_dummies(dados_codificados[colunas_dummies], prefix=colunas_dummies)
        dados_codificados = pd.concat([dados_codificados, dummies], axis=1)
        dados_codificados = dados_codificados.drop(columns=colunas_dummies)
        st.success(f"🏷️ One-Hot Encoding aplicado em: {', '.join(colunas_dummies)}")

    return dados_codificados


# Aplicar as duas codificações
df_codificado = codificar_variaveis_completo(df, COLUNAS_LABEL, COLUNAS_DUMMIES)

# Mostrar resultado
st.dataframe(df_codificado.head())

# Mensagem de sucesso
st.markdown("<p style='color:blue; font-weight:bold;'>✅ Dados codificados com sucesso (LabelEncoder + Dummies)!</p>",
            unsafe_allow_html=True)


# =====================================================================
# 6️⃣ Salvar dataset pré-processado
# =====================================================================

# Criar pasta outputs/data se não existir
output_dir = os.path.join("outputs", "data")
os.makedirs(output_dir, exist_ok=True)

# Caminho completo do arquivo
output_path = os.path.join(output_dir, "dataset_preprocessado.csv")

# Salvar arquivo
df_codificado.to_csv(output_path, index=False, encoding='utf-8')
st.success(f"💾 Dataset pré-processado salvo em: `{output_path}`")