import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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

    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2, palette=pie_colors, legend=False)
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
        st.success(f"💾 Gráfico salvo em: `{fig_path}`")

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
