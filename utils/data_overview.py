
import streamlit as st
import streamlit as st
from io import StringIO


def data_overview(df):
    """
    Exibe uma visão geral do dataset no Streamlit.
    """

    if df is None or df.empty:
        st.warning("⚠️ Nenhum dado disponível para exibir.")
        return

    st.markdown("## Visão Geral do Dataset")
    st.write("---")

    # Mostrar cabeçalho e rodapé
    st.subheader("👀 Amostra do Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Primeiras linhas:**")
        st.dataframe(df.head())
    with col2:
        st.markdown("**Últimas linhas:**")
        st.dataframe(df.tail())

    # Shape
    st.write("---")
    st.subheader("📏 Dimensões do Dataset")
    st.write(f"- **Total de linhas:** {df.shape[0]}")
    st.write(f"- **Total de colunas:** {df.shape[1]}")

    # Info geral
    buffer = StringIO()              # cria um "arquivo em memória"
    df.info(buf=buffer)              # escreve o resultado dentro do buffer
    info_str = buffer.getvalue()     # extrai o conteúdo como string

    st.text("📋 Informações do DataFrame:")
    st.text(info_str)

    # Estatísticas descritivas
    st.write("---")
    st.subheader("📈 Estatísticas Descritivas")
    st.dataframe(df.describe().T)

    # Colunas categóricas e numéricas
    st.write("---")
    st.subheader("🔠 Tipos de Colunas")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    st.write(f"**Colunas Categóricas:** {cat_cols}")
    st.write(f"**Colunas Numéricas:** {num_cols}")

    # Valores nulos
    st.write("---")
    st.subheader("🚫 Valores Nulos")
    st.dataframe(
        df.isnull().sum().reset_index().rename(columns={"index": "Coluna", 0: "Nulos"})
    )

    # Duplicatas
    st.write("---")
    st.subheader("🧩 Verificação de Duplicatas")
    if df.duplicated().any():
        st.error("⚠️ Existem linhas duplicadas no dataset.")
    else:
        st.success("✅ Nenhuma linha duplicada encontrada.")

    st.write("---")
    st.markdown(
        "🎯 **Análise inicial concluída!** Explore as próximas seções para visualizações e insights."
    )


