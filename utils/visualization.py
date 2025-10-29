import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import skew

# Paleta de cores padrão
palette = ['#CBCE91FF', '#EA738DFF']
color_palette = sns.color_palette(palette)

# =====================================================================
# Função: gráfico de distribuição (pizza + barras)
# =====================================================================
def single_plot_distribution(column_name, dataframe, save_fig=True):
    """
    Mostra e salva um gráfico de distribuição (pizza + barras) para uma coluna categórica.
    """
    value_counts = dataframe[column_name].value_counts()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1]}
    )

    pie_colors = color_palette[:len(value_counts)]
    wedges, texts, autotexts = ax1.pie(
        value_counts,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
        colors=pie_colors,
        labels=value_counts.index
    )
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    ax1.add_artist(centre_circle)
    ax1.set_title(f"Distribuição de {column_name}", fontsize=14)

    sns.barplot(
    x=value_counts.index,
    y=value_counts.values,
    hue=value_counts.index,  
    legend=False,            
    ax=ax2,
    palette=pie_colors
)
    ax2.set_title(f"Contagem de {column_name}", fontsize=14)
    ax2.set_xlabel(column_name)
    ax2.set_ylabel("Frequência")
    ax2.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    st.pyplot(fig)

    if save_fig:
        save_dir = os.path.join("outputs", "figures")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, f"{column_name}_distribuicao.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        #st.success(f"💾 Gráfico salvo em: `{fig_path}`")

    plt.close(fig)

# =====================================================================
# Função: scatter plot avançado
# =====================================================================
def advanced_scatter_plot(x_column, y_column, target_column, dataframe, save_fig=True):
    """
    Mostra e salva um gráfico de dispersão entre duas variáveis com cor baseada em uma terceira variável.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=x_column,
        y=y_column,
        hue=target_column,
        data=dataframe,
        palette=color_palette,
        s=80,
        edgecolor="white",
        alpha=0.8,
        ax=ax
    )

    ax.set_title(f"{x_column} vs {y_column} (por {target_column})", fontsize=14)
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.grid(True)
    plt.tight_layout()

    # Exibir no Streamlit
    st.pyplot(fig)

    # Salvar figura
    if save_fig:
        save_dir = os.path.join("outputs", "figures")
        os.makedirs(save_dir, exist_ok=True)
        fig_name = f"scatter_{x_column}_vs_{y_column}_por_{target_column}.png"
        fig_path = os.path.join(save_dir, fig_name)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        st.success(f"💾 Gráfico salvo em: `{fig_path}`")

    plt.close(fig)
    
    # =====================================================================
# Função: pairplot (relações entre variáveis)
# =====================================================================
def plot_pairplot(dataframe, columns_to_plot, target_column, save_fig=True):
    """
    Gera e salva um pairplot mostrando relações entre múltiplas variáveis numéricas
    coloridas pela variável alvo.

    Parâmetros:
        dataframe (pd.DataFrame): Dataset de entrada.
        columns_to_plot (list): Lista de colunas a incluir no pairplot.
        target_column (str): Nome da variável alvo (usada para colorir os pontos).
        save_fig (bool): Se True, salva o gráfico em 'outputs/figures/'.
    """
    #st.subheader("Relações entre variáveis (Pairplot)")

    # Filtra apenas as colunas desejadas
    data_to_plot = dataframe[columns_to_plot].copy()

    # Define o mapa de cores para as classes
    unique_classes = dataframe[target_column].unique()
    Q_colors = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(unique_classes)}

    # Cria o pairplot
    pairplot_fig = sns.pairplot(data_to_plot, hue=target_column, palette=Q_colors, diag_kind="kde")

    # Ajusta layout
    pairplot_fig.fig.suptitle("Pairplot das Variáveis Selecionadas", fontsize=14, y=1.02)

    # Exibe no Streamlit
    st.pyplot(pairplot_fig)

    # Salva o gráfico
    if save_fig:
        save_dir = os.path.join("outputs", "figures")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "pairplot_variaveis.png")
        pairplot_fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        #st.success(f"💾 Gráfico salvo em: `{fig_path}`")

    plt.close()

# ===================================================================================================
# Função: detecção de outliers (Boxplots múltiplos)
# ===================================================================================================
def plot_outlier_detection(dataframe, save_fig=True):
    """
    Gera boxplots de todas as colunas numéricas para identificar outliers,
    exibe no Streamlit e salva o gráfico.

    Parâmetros:
        dataframe (pd.DataFrame): Dataset tratado.
        save_fig (bool): Se True, salva a figura em 'outputs/figures/'.
    """
    # Seleciona colunas numéricas
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    if not num_cols:
        st.warning("⚠️ Nenhuma coluna numérica encontrada para detecção de outliers.")
        return

    # Define o layout da grade
    num_rows = 3
    num_cols_grid = 3

    # Cria a figura
    fig, axes = plt.subplots(num_rows, num_cols_grid, figsize=(25, 17))
    axes = axes.flatten()

    palette = sns.color_palette(['#CBCE91FF', '#EA738DFF'])

    # Cria os boxplots
    for i, col in enumerate(num_cols[:num_rows * num_cols_grid]):
        sns.boxplot(x=dataframe[col], ax=axes[i], color=palette[i % len(palette)])
        axes[i].set_title(col, fontsize=14)
        axes[i].grid(True)

    # Remove os subplots vazios
    for i in range(len(num_cols), num_rows * num_cols_grid):
        fig.delaxes(axes[i])

    plt.tight_layout()

    # Exibe no Streamlit
    st.pyplot(fig)

    # Salvar figura
    if save_fig:
        save_dir = os.path.join("outputs", "figures")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "detecao_outliers.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        #st.success(f"💾 Gráfico salvo em: `{fig_path}`")

    plt.close(fig)
    
# # # =====================================================================================================
# # #                        Function to Plot Numerical Features
# #====================================================================================================

def plot_numerical_distribution_with_hue(dataframe, hue_col='Qualidade', save_fig=True):
    """
    Plota distribuições numéricas coloridas por uma variável categórica (hue).
    Exibe os gráficos no Streamlit e salva a figura.

    Parâmetros:
        dataframe (pd.DataFrame): Dataset tratado.
        hue_col (str): Coluna categórica usada para colorir os gráficos.
        save_fig (bool): Se True, salva a figura em 'outputs/figures/'.
    """
    # Selecionar colunas numéricas
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    if not num_cols:
        st.warning("⚠️ Nenhuma coluna numérica encontrada para visualização.")
        return

    palette = sns.color_palette(['#CBCE91FF', '#EA738DFF'])

    # Definir número de linhas e colunas
    rows = (len(num_cols) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(20, rows * 5), dpi=120)
    axes = axes.flatten()

    # Loop para criar os gráficos
    for i, col in enumerate(num_cols):
        sns.histplot(data=dataframe, x=col, hue=hue_col, kde=True, ax=axes[i], palette=palette)
        axes[i].set_title(f"Distribuição de {col}", fontsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

        # Calcular assimetria (skewness)
        skewness_value = skew(dataframe[col].dropna())
        axes[i].annotate(f"Skewness: {skewness_value:.2f}",
                         xy=(0.05, 0.9), xycoords='axes fraction',
                         fontsize=12, color='red')

    # Remover subplots vazios
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Exibir no Streamlit
    st.pyplot(fig)

    # Salvar a figura
    if save_fig:
        save_dir = os.path.join("outputs", "figures")
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "numerical_distribution.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        #st.success(f"💾 Gráfico salvo em: `{fig_path}`")

    plt.close(fig)
