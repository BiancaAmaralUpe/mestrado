# utils.py

from pathlib import Path
import pandas as pd

# ==============================================================================
# Diretórios principais do projeto
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_RELATORIOS_DIR = BASE_DIR / "output-relatorios"


def criar_diretorios_necessarios() -> None:
    """
    Cria as pastas usadas pelo projeto, caso ainda não existam.

    Para este exercício, será criada a pasta:
        - output-relatorios

    Essa pasta será usada para salvar:
        - relatório final em .txt
        - log da execução
    """

    diretorios = [
        OUTPUT_RELATORIOS_DIR,
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)


def salvar_relatorio_random_forest_txt(
    df_resultados: pd.DataFrame,
    caminho_relatorio: Path,
) -> None:
    """
    Salva em arquivo .txt o resumo final dos resultados dos modelos.

    O relatório contém:
        - tabela comparativa dos modelos
        - legenda das métricas
        - identificação automática dos principais resultados
    """

    linhas = []

    linhas.append("=" * 120)
    linhas.append("RESUMO FINAL DOS RESULTADOS")
    linhas.append("=" * 120)
    linhas.append("")

    linhas.append("TABELA COMPARATIVA DOS RESULTADOS")
    linhas.append("-" * 120)

    cabecalho = (
        f"{'Modelo':<35} "
        f"{'Acurácia Treino (%)':<22} "
        f"{'Acurácia Teste (%)':<21} "
        f"{'Gap (%)':<12} "
        f"{'Tempo (s)':<12}"
    )

    linhas.append(cabecalho)
    linhas.append("-" * 120)

    for _, resultado in df_resultados.iterrows():
        linha = (
            f"{resultado['Modelo']:<35} "
            f"{resultado['Acurácia Treino (%)']:<22.2f} "
            f"{resultado['Acurácia Teste (%)']:<21.2f} "
            f"{resultado['Gap (%)']:<12.2f} "
            f"{resultado['Tempo de Treinamento (s)']:<12.4f}"
        )

        linhas.append(linha)

    linhas.append("")
    linhas.append("=" * 120)
    linhas.append("LEGENDA")
    linhas.append("=" * 120)
    linhas.append("Acurácia Treino (%) = percentual de acertos no conjunto de treino.")
    linhas.append("Acurácia Teste (%)  = percentual de acertos no conjunto de teste.")
    linhas.append("Gap (%)             = diferença entre acurácia de treino e acurácia de teste.")
    linhas.append("Tempo (s)           = tempo gasto para treinar o modelo.")
    linhas.append("")

    linhas.append("=" * 120)
    linhas.append("DESTAQUES DOS RESULTADOS")
    linhas.append("=" * 120)

    modelo_maior_gap = df_resultados.sort_values(
        by="Gap (%)",
        ascending=False
    ).iloc[0]

    modelo_menor_gap = df_resultados.sort_values(
        by="Gap (%)",
        ascending=True
    ).iloc[0]

    modelo_melhor_teste = df_resultados.sort_values(
        by="Acurácia Teste (%)",
        ascending=False
    ).iloc[0]

    linhas.append(
        f"Modelo com maior Gap: {modelo_maior_gap['Modelo']} "
        f"({modelo_maior_gap['Gap (%)']:.2f}%)."
    )

    linhas.append(
        f"Modelo com menor Gap: {modelo_menor_gap['Modelo']} "
        f"({modelo_menor_gap['Gap (%)']:.2f}%)."
    )

    linhas.append(
        f"Modelo com maior acurácia de teste: {modelo_melhor_teste['Modelo']} "
        f"({modelo_melhor_teste['Acurácia Teste (%)']:.2f}%)."
    )

    linhas.append("")

    caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_relatorio, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))

    print(f"\nRelatório TXT salvo em: {caminho_relatorio}")


class Tee:
    """
    Duplica a saída do terminal para um arquivo.

    Tudo que for impresso com print() continuará aparecendo no terminal
    e também será salvo no arquivo de log.
    """

    def __init__(self, *arquivos):
        self.arquivos = arquivos

    def write(self, texto):
        for arquivo in self.arquivos:
            arquivo.write(texto)
            arquivo.flush()

    def flush(self):
        for arquivo in self.arquivos:
            arquivo.flush()