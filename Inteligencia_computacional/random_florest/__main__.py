# __main__.py

# __main__.py

import sys
import pandas as pd

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.base.carregar_dados import carregar_dataset

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.base.limpeza_dados import verificar_dados, limpar_dados

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.base.preparacao_dados import separar_treino_teste

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.tipos_random_florest.arvore_solitaria import avaliar_arvore_solitaria, treinar_arvore_solitaria

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.tipos_random_florest.pequena_floresta import treinar_pequena_floresta, avaliar_pequena_floresta

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.tipos_random_florest.grande_floresta_paralelizada import treinar_grande_floresta_paralelizada, avaliar_grande_floresta_paralelizada

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.tipos_random_florest.floresta_podada import treinar_floresta_podada, avaliar_floresta_podada

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.random_florest.utils import OUTPUT_RELATORIOS_DIR, criar_diretorios_necessarios, salvar_relatorio_random_forest_txt, Tee

# ============================================================================ #
# como executar:
# python -m Inteligencia_computacional.random_florest
# ============================================================================ #
def exibir_cabecalho(titulo: str) -> None:
    """
    Exibe um cabeçalho visual no terminal.
    """

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def adicionar_tempo_resultado(resultado: dict, tempo_treinamento: float) -> dict:
    """
    Adiciona o tempo de treinamento ao dicionário de resultado do modelo.
    """

    resultado["tempo_treinamento"] = tempo_treinamento

    return resultado


def exibir_tabela_comparativa(resultados: list) -> pd.DataFrame:
    """
    Exibe a tabela comparativa final com os resultados dos modelos
    e retorna o DataFrame formatado para salvar no relatório.
    """

    exibir_cabecalho("TABELA COMPARATIVA FINAL")

    df_resultados = pd.DataFrame(resultados)

    # ==============================================================================
    # Conversão das métricas para percentual.
    #
    # As funções de avaliação retornam acurácia e gap em formato decimal.
    # Exemplo:
    # 0.85 representa 85%.
    #
    # Para o relatório, é mais fácil visualizar esses valores em percentual.
    # ==============================================================================
    df_resultados["Acurácia Treino (%)"] = (
        df_resultados["acuracia_treino"] * 100
    ).round(2)

    df_resultados["Acurácia Teste (%)"] = (
        df_resultados["acuracia_teste"] * 100
    ).round(2)

    df_resultados["Gap (%)"] = (
        df_resultados["gap"] * 100
    ).round(2)

    df_resultados["Tempo de Treinamento (s)"] = (
        df_resultados["tempo_treinamento"]
    ).round(4)

    # ==============================================================================
    # Seleção e organização das colunas finais da tabela.
    # ==============================================================================
    tabela_final = df_resultados[
        [
            "modelo",
            "Acurácia Treino (%)",
            "Acurácia Teste (%)",
            "Gap (%)",
            "Tempo de Treinamento (s)"
        ]
    ]

    tabela_final = tabela_final.rename(columns={
        "modelo": "Modelo"
    })

    print(tabela_final.to_string(index=False))

    return tabela_final


def executar_experimento() -> None:
    """
    Executa o experimento completo da missão prática.
    """

    exibir_cabecalho("RESULTADO")

    # ==============================================================================
    # Criação dos diretórios necessários
    # ==============================================================================
    criar_diretorios_necessarios()

    # ==============================================================================
    # Carregamento do dataset
    # ==============================================================================
    df = carregar_dataset("diabetes.csv")

    # ==============================================================================
    # Verificação inicial dos dados
    # ==============================================================================
    verificar_dados(df)

    # ==============================================================================
    # Limpeza dos dados
    #
    # Remove duplicadas e trata zeros suspeitos nas colunas clínicas,
    # conforme definido no arquivo limpeza_dados.py.
    # ==============================================================================
    df_limpo = limpar_dados(df)

    # ==============================================================================
    # Separação entre treino e teste
    #
    # test_size=0.3
    # random_state=42
    #
    # Essa configuração está dentro da função separar_treino_teste.
    # ==============================================================================
    X_train, X_test, y_train, y_test = separar_treino_teste(df_limpo)

    resultados = []

    # ==============================================================================
    # Modelo 1 - Árvore Solitária
    #
    # DecisionTreeClassifier(random_state=42)
    # ==============================================================================
    modelo_arvore, tempo_arvore = treinar_arvore_solitaria(
        X_train,
        y_train
    )

    resultado_arvore = avaliar_arvore_solitaria(
        modelo_arvore,
        X_train,
        X_test,
        y_train,
        y_test
    )

    resultado_arvore = adicionar_tempo_resultado(
        resultado_arvore,
        tempo_arvore
    )

    resultados.append(resultado_arvore)

    # ==============================================================================
    # Modelo 2 - Pequena Floresta
    #
    # RandomForestClassifier(n_estimators=10, random_state=42)
    # ==============================================================================
    modelo_pequena_floresta, tempo_pequena_floresta = treinar_pequena_floresta(
        X_train,
        y_train
    )

    resultado_pequena_floresta = avaliar_pequena_floresta(
        modelo_pequena_floresta,
        X_train,
        X_test,
        y_train,
        y_test
    )

    resultado_pequena_floresta = adicionar_tempo_resultado(
        resultado_pequena_floresta,
        tempo_pequena_floresta
    )

    resultados.append(resultado_pequena_floresta)

    # ==============================================================================
    # Modelo 3 - Grande Floresta Paralelizada
    #
    # RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    # ==============================================================================
    modelo_grande_floresta, tempo_grande_floresta = (
        treinar_grande_floresta_paralelizada(
            X_train,
            y_train
        )
    )

    resultado_grande_floresta = avaliar_grande_floresta_paralelizada(
        modelo_grande_floresta,
        X_train,
        X_test,
        y_train,
        y_test
    )

    resultado_grande_floresta = adicionar_tempo_resultado(
        resultado_grande_floresta,
        tempo_grande_floresta
    )

    resultados.append(resultado_grande_floresta)

    # ==============================================================================
    # Modelo 4 - Floresta Podada
    #
    # RandomForestClassifier(
    #     n_estimators=150,
    #     max_depth=5,
    #     n_jobs=-1,
    #     random_state=42
    # )
    # ==============================================================================
    modelo_floresta_podada, tempo_floresta_podada = treinar_floresta_podada(
        X_train,
        y_train
    )

    resultado_floresta_podada = avaliar_floresta_podada(
        modelo_floresta_podada,
        X_train,
        X_test,
        y_train,
        y_test
    )

    resultado_floresta_podada = adicionar_tempo_resultado(
        resultado_floresta_podada,
        tempo_floresta_podada
    )

    resultados.append(resultado_floresta_podada)

    # ==============================================================================
    # Tabela comparativa final
    #
    # Esta é a tabela que será usada para responder ao relatório.
    # ==============================================================================
    df_resultados = exibir_tabela_comparativa(resultados)

    # ==============================================================================
    # Salvamento do relatório TXT
    # ==============================================================================
    caminho_txt = OUTPUT_RELATORIOS_DIR / "relatorio_random_forest.txt"

    salvar_relatorio_random_forest_txt(
        df_resultados=df_resultados,
        caminho_relatorio=caminho_txt
    )

    exibir_cabecalho("EXPERIMENTO CONCLUÍDO")
    print("Todos os modelos foram treinados, avaliados e salvos com sucesso.")


if __name__ == "__main__":
    criar_diretorios_necessarios()

    caminho_log = OUTPUT_RELATORIOS_DIR / "log_execucao_random_forest.txt"

    with open(caminho_log, "w", encoding="utf-8") as arquivo_log:
        terminal_original = sys.stdout
        sys.stdout = Tee(sys.stdout, arquivo_log)

        try:
            executar_experimento()
        finally:
            sys.stdout = terminal_original

    print(f"\nLog da execução salvo em: {caminho_log}")