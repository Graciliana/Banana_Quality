# =================================================================================================
# 1_📊_EDA.PY — Página de Análise Exploratória de Dados
# =================================================================================================

from matplotlib import pyplot as plt
import streamlit as st
import seaborn as sns 
import sys
import os


from utils.data_overview import data_overview
from utils.data_processing import carregar_dataset_zip, mapear_qualidade, renomear_colunas
from utils.visualization import plot_numerical_distribution_with_hue, plot_outlier_detection, plot_pairplot, single_plot_distribution

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# =================================================================================================
# CONFIGURAÇÃO
# =================================================================================================
st.set_page_config(
    page_title="EDA - Qaulidade Banana ",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análise Exploratória de Dados (EDA)")

# =================================================================================================
# CARREGAR DATASET
# =================================================================================================
# Caminho do arquivo ZIP
zip_path = "./Data/banana_quality.zip"

# Carregar dataset
df = carregar_dataset_zip(zip_path)
# mostrar o datasets
df = renomear_colunas(df)
df = mapear_qualidade(df)
df = carregar_dataset_zip(zip_path)
if df is not None:
    df = renomear_colunas(df)
    df = mapear_qualidade(df)
    #data_overview(df)
else:
    st.warning("⚠️ Não foi possível carregar o dataset.")
# =================================================================================================
# Distribuição de Variáveis Categóricas
# =================================================================================================

# Mostra gráficos de distribuição
st.subheader("Distribuição de Variáveis Categóricas")
single_plot_distribution("Qualidade", df)

# =================================================================================================
# Relações entre variáveis (Pairplot)
# =================================================================================================
# Pairplot de variáveis
st.subheader("Relações entre variáveis (Pairplot)")
columns_to_plot = ['Tamanho', 'Peso', 'Doçura', 'Maciez', 'Época de Colheita',
                   'Maturidade', 'Acidez', 'Qualidade']
plot_pairplot(df, columns_to_plot, target_column='Qualidade')

# =================================================================================================
# Dectção de outliers
# =================================================================================================
##### Dectção de outliers########
st.subheader("Detecção de Outliers")
plot_outlier_detection(df)

# =================================================================================================
# Distribuição de Variáveis Numéricas
# =================================================================================================
##### Distribuição de Variáveis Numéricas ########
st.subheader("Distribuição de Variáveis Numéricas com Hue (Qualidade)")
plot_numerical_distribution_with_hue(df)

##### conclusão ########
st.subheader("Alguns Insights")
st.text("Observado que os dados estão em estado gaussiano")