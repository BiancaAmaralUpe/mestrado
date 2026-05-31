# ======================================================================================== #
# Responsável por orquestrar toda a prática XOR.
# ======================================================================================== #

# pyrefly: ignore [missing-import]
import pandas as pd

# pyrefly: ignore [missing-import]
from .dados_xor import (
    carregar_dados_xor,
    criar_tabela_xor,
    exibir_tabela_xor,
)

# pyrefly: ignore [missing-import]
from .treinamento import (
    treinar_perceptron_simples,
    criar_historico_perceptron_df,
    treinar_mlp,
)

# pyrefly: ignore [missing-import]
from .avaliacao import (
    avaliar_predicoes,
    gerar_matriz_confusao,
    criar_tabela_resultados_perceptron,
    criar_tabela_resultados_mlp,
    criar_comparacao_final,
    exibir_resultados_perceptron,
    exibir_resultados_mlp,
    exibir_comparacao_final,
    exibir_conclusao_final,
)

# pyrefly: ignore [missing-import]
from .graficos import (
    plotar_pontos_xor,
    plotar_historico_perceptron,
    plotar_fronteira_decisao,
    plotar_matriz_confusao,
    plotar_representacao_oculta_mlp,
)

# pyrefly: ignore [missing-import]
from .utils import (
    definir_semente,
    criar_diretorios_necessarios,
    executar_com_log,
)


# =================================================================================== #
# Para executar o módulo:
# python -m Inteligencia_computacional.pratica_mlp.pratica_xor
# =================================================================================== #


def main():
    """
    Executa o fluxo completo da prática XOR.

    A prática compara:
        - Perceptron simples;
        - MLP.

    O objetivo é mostrar que:
        - o Perceptron simples não resolve XOR perfeitamente;
        - a MLP resolve XOR por usar uma camada oculta não linear.
    """

    # ======================================================================================== #
    # Configurações iniciais.
    #
    # A seed ajuda a manter o resultado reprodutível.
    # Os diretórios são criados para salvar imagens e relatórios.
    # ======================================================================================== #
    definir_semente()
    criar_diretorios_necessarios()

    print("\n" + "=" * 80)
    print("PRÁTICA XOR - PERCEPTRON SIMPLES VS MLP")
    print("=" * 80)

    # ======================================================================================== #
    # Carregamento dos dados XOR.
    #
    # O XOR possui apenas quatro combinações possíveis:
    #   0 XOR 0 = 0
    #   0 XOR 1 = 1
    #   1 XOR 0 = 1
    #   1 XOR 1 = 0
    # ======================================================================================== #
    X, y = carregar_dados_xor()

    tabela_xor = criar_tabela_xor(
        X=X,
        y=y,
    )

    exibir_tabela_xor(tabela_xor)

    # ======================================================================================== #
    # Visualização inicial dos pontos XOR.
    #
    # Esse gráfico mostra que as classes estão posicionadas em diagonais opostas.
    # Isso explica por que uma única reta não consegue separar o XOR perfeitamente.
    # ======================================================================================== #
    print("\nGerando gráfico dos pontos XOR...")

    plotar_pontos_xor(
        X=X,
        y=y,
        titulo="XOR dataset: classes em diagonais opostas",
    )

    # ======================================================================================== #
    # Treinamento do Perceptron simples.
    #
    # O Perceptron simples tenta resolver o XOR usando apenas uma fronteira linear.
    # Como o XOR não é linearmente separável, espera-se que ele não atinja 100% de acerto.
    # ======================================================================================== #
    perceptron = treinar_perceptron_simples(
        X=X,
        y=y,
    )

    historico_perceptron_df = criar_historico_perceptron_df(
        perceptron=perceptron,
    )

    print("\nHistórico do Perceptron simples:")
    print("-" * 70)
    print(
        historico_perceptron_df[
            [
                "epoch",
                "errors_in_epoch",
                "accuracy_after_epoch",
            ]
        ].to_string(index=False)
    )

    # ======================================================================================== #
    # Gráfico da acurácia do Perceptron por época.
    #
    # Esse gráfico ajuda a visualizar que o Perceptron não converge para solução perfeita.
    # ======================================================================================== #
    print("\nGerando gráfico do histórico do Perceptron...")

    plotar_historico_perceptron(
        historico_df=historico_perceptron_df,
    )

    # ======================================================================================== #
    # Avaliação do melhor Perceptron encontrado.
    #
    # Mesmo que o Perceptron não resolva XOR perfeitamente, guardamos os melhores pesos
    # encontrados durante o treinamento para visualizar a melhor reta obtida.
    # ======================================================================================== #
    melhores_pesos = perceptron.melhores_pesos_
    melhor_bias = perceptron.melhor_bias_

    def score_perceptron(pontos):
        """
        Calcula o score linear do Perceptron:
            z = Xw + b
        """

        return pontos @ melhores_pesos + melhor_bias

    def prever_perceptron(pontos):
        """
        Converte o score linear do Perceptron em classe 0 ou 1.
        """

        return (score_perceptron(pontos) >= 0).astype(int)

    predicoes_perceptron = prever_perceptron(X)

    acuracia_perceptron = avaliar_predicoes(
        y_real=y,
        y_previsto=predicoes_perceptron,
    )

    tabela_resultados_perceptron = criar_tabela_resultados_perceptron(
        tabela_xor=tabela_xor,
        predicoes_perceptron=predicoes_perceptron,
    )

    exibir_resultados_perceptron(
        pesos=melhores_pesos,
        bias=melhor_bias,
        acuracia=acuracia_perceptron,
        tabela_resultados=tabela_resultados_perceptron,
    )

    # ======================================================================================== #
    # Fronteira de decisão do Perceptron simples.
    #
    # Como o Perceptron é linear, a fronteira será uma linha reta.
    # Essa visualização mostra por que ele falha no XOR.
    # ======================================================================================== #
    print("\nGerando fronteira de decisão do Perceptron simples...")

    plotar_fronteira_decisao(
        X=X,
        y=y,
        funcao_predicao=prever_perceptron,
        funcao_score=score_perceptron,
        threshold=0.0,
        titulo="Perceptron simples no XOR: fronteira linear",
        nome_arquivo="fronteira_perceptron.png",
    )

    # ======================================================================================== #
    # Treinamento da MLP.
    #
    # A MLP possui camada oculta com ativação não linear.
    # Isso permite transformar o espaço de entrada e resolver o XOR.
    # ======================================================================================== #
    modelo_mlp = treinar_mlp(
        X=X,
        y=y,
    )

    predicoes_mlp = modelo_mlp.predict(X)
    probabilidades_mlp = modelo_mlp.predict_proba(X)[:, 1]

    acuracia_mlp = avaliar_predicoes(
        y_real=y,
        y_previsto=predicoes_mlp,
    )

    tabela_resultados_mlp = criar_tabela_resultados_mlp(
        tabela_xor=tabela_xor,
        probabilidades_mlp=probabilidades_mlp,
        predicoes_mlp=predicoes_mlp,
    )

    exibir_resultados_mlp(
        acuracia=acuracia_mlp,
        tabela_resultados=tabela_resultados_mlp,
    )

    # ======================================================================================== #
    # Fronteira de decisão da MLP.
    #
    # A fronteira é desenhada usando a probabilidade da classe 1.
    # A região de decisão muda quando P(classe 1) passa de 0.5.
    # ======================================================================================== #
    print("\nGerando fronteira de decisão da MLP...")

    def score_mlp(pontos):
        """
        Retorna a probabilidade da classe 1 estimada pela MLP.
        """

        return modelo_mlp.predict_proba(pontos)[:, 1]

    def prever_mlp(pontos):
        """
        Retorna a classe prevista pela MLP.
        """

        return modelo_mlp.predict(pontos)

    plotar_fronteira_decisao(
        X=X,
        y=y,
        funcao_predicao=prever_mlp,
        funcao_score=score_mlp,
        threshold=0.5,
        titulo="MLP no XOR: fronteira não linear",
        nome_arquivo="fronteira_mlp.png",
    )

    # ======================================================================================== #
    # Representação interna da camada oculta da MLP.
    #
    # Essa visualização mostra como a camada oculta transforma os pontos do XOR
    # antes da classificação final.
    # ======================================================================================== #
    print("\nGerando representação da camada oculta da MLP...")

    plotar_representacao_oculta_mlp(
        X=X,
        y=y,
        modelo_mlp=modelo_mlp,
    )

    # ======================================================================================== #
    # Matrizes de confusão.
    #
    # As matrizes mostram a diferença entre os modelos:
    #   - o Perceptron simples comete erro;
    #   - a MLP deve acertar todos os quatro casos do XOR.
    # ======================================================================================== #
    matriz_confusao_perceptron = gerar_matriz_confusao(
        y_real=y,
        y_previsto=predicoes_perceptron,
    )

    matriz_confusao_mlp = gerar_matriz_confusao(
        y_real=y,
        y_previsto=predicoes_mlp,
    )

    print("\nGerando matriz de confusão do Perceptron simples...")

    plotar_matriz_confusao(
        matriz_confusao=matriz_confusao_perceptron,
        titulo="Matriz de confusão - Perceptron simples",
        nome_arquivo="matriz_confusao_perceptron.png",
    )

    print("\nGerando matriz de confusão da MLP...")

    plotar_matriz_confusao(
        matriz_confusao=matriz_confusao_mlp,
        titulo="Matriz de confusão - MLP",
        nome_arquivo="matriz_confusao_mlp.png",
    )

    # ======================================================================================== #
    # Comparação final.
    #
    # Esta tabela resume a diferença principal:
    #   - Perceptron simples: fronteira linear;
    #   - MLP: fronteira não linear.
    # ======================================================================================== #
    comparacao = criar_comparacao_final(
        acuracia_perceptron=acuracia_perceptron,
        acuracia_mlp=acuracia_mlp,
    )

    exibir_comparacao_final(
        comparacao=comparacao,
    )

    exibir_conclusao_final(
        acuracia_perceptron=acuracia_perceptron,
        acuracia_mlp=acuracia_mlp,
    )


if __name__ == "__main__":
    executar_com_log(main)