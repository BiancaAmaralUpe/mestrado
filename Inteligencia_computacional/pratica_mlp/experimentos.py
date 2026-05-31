# pyrefly: ignore [missing-import]
import torch.optim as optim

from .constants import (
    RANDOM_SEED,
    QUANTIDADE_PIXELS_IMAGEM,
    QUANTIDADE_DIGITOS,
)

from .carregar_dados import executar_carregamento_dados
from .modelo_mlp import ModeloMLP, contar_parametros_treinaveis
from .treinamento import treinar_modelo
from .avaliacao_dados import (
    carregar_melhor_modelo,
    avaliar_modelo,
    coletar_predicoes,
    gerar_matriz_confusao,
    exibir_relatorio_avaliacao_final,
)
from .graficos_dados import (
    mostrar_exemplos_mnist,
    plotar_historico_treinamento,
    plotar_matriz_confusao,
    mostrar_predicoes_modelo,
)
from .funcoes_perda import criar_funcao_perda
from .utils import criar_diretorios_experimento


# ================================================================================================= #
# Executa um experimento completo da MLP no MNIST.
# ================================================================================================= #

def executar_experimento(configuracao: dict, dispositivo):
    """
    Executa um experimento completo da MLP no MNIST.

    Cada experimento pode alterar:
        - quantidade de épocas;
        - tamanho do batch;
        - taxa de aprendizado;
        - percentual de validação;
        - tamanho da camada oculta;
        - função de perda.
    """

    nome_experimento = configuracao["nome"]

    print("\n" + "=" * 80)
    print(f"INICIANDO EXPERIMENTO: {nome_experimento}")
    print("=" * 80)

    caminho_checkpoint, diretorio_imagens = criar_diretorios_experimento(
        nome_experimento=nome_experimento
    )

    train_loader, val_loader, test_loader = executar_carregamento_dados(
        tamanho_batch=configuracao["tamanho_batch"],
        percentual_validacao=configuracao["percentual_validacao"],
        random_seed=RANDOM_SEED,
    )

    mostrar_exemplos_mnist(
        data_loader=train_loader,
        quantidade_imagens=16,
        diretorio_saida=diretorio_imagens,
    )

    print("\nCriando modelo MLP...")
    # ================================================================================================= #
    # criação do modelo MLP
    # ================================================================================================= #   
    # Quantos valores entram no modelo;
    # Quantos neurônios terá a camada oculta;
    # Quantas classes o modelo precisa prever
    # ================================================================================================= #
    modelo = ModeloMLP(
        quantidade_pixels_imagem=QUANTIDADE_PIXELS_IMAGEM,
        quantidade_neuronios_ocultos=configuracao["quantidade_neuronios_ocultos"],
        quantidade_digitos=QUANTIDADE_DIGITOS,
    ).to(dispositivo)

    print(modelo)
    print(f"Parâmetros treináveis: {contar_parametros_treinaveis(modelo):,}")

    funcao_perda = criar_funcao_perda(configuracao["tipo_loss"])
    # ================================================================================================= #
    # O otimizador é quem ajusta os pesos do modelo.
    # O Adam é um otimizador muito usado em redes neurais porque ajusta os pesos de forma eficiente. 
    # Ele adapta o tamanho dos passos para cada parâmetro do modelo.
    # ================================================================================================= #
    otimizador = optim.Adam(
        modelo.parameters(), # entrega para o otimizador todos os parâmetros treináveis da MLP
        lr=configuracao["taxa_aprendizado"], # lr significa learning rate
    )

    print("\n==============Iniciando treinamento===============")
    # ================================================================================================= #
    # Ela executa o ciclo completo de treinamento.
    # A função treinar_modelo percorre todas as épocas definidas na configuração.
    # Em cada época, o modelo treina com train_loader e é avaliado com val_loader.
    # Durante esse processo, são armazenadas as métricas de loss e acurácia
    # de treino e validação.
    # O melhor modelo é salvo em checkpoint com base na maior acurácia de validação.
    # O histórico retornado será usado depois para gerar os gráficos de treinamento.
    # ================================================================================================= #
    historico = treinar_modelo(
        modelo=modelo,
        train_loader=train_loader,
        val_loader=val_loader,
        funcao_perda=funcao_perda,
        otimizador=otimizador,
        quantidade_epocas=configuracao["quantidade_epocas"],
        caminho_checkpoint=caminho_checkpoint,
        dispositivo=dispositivo,
        quantidade_pixels_imagem=QUANTIDADE_PIXELS_IMAGEM,
        quantidade_neuronios_ocultos=configuracao["quantidade_neuronios_ocultos"],
        quantidade_digitos=QUANTIDADE_DIGITOS,
        taxa_aprendizado=configuracao["taxa_aprendizado"],
        tamanho_batch=configuracao["tamanho_batch"],
    )

    print("\nGerando gráficos de treinamento...")
    # ================================================================================================= #
    # Ela pega os valores salvos durante o treinamento e cria dois gráficos
    # ================================================================================================= #
    plotar_historico_treinamento(
        historico=historico,
        diretorio_saida=diretorio_imagens,
    )

    print("\nCarregando melhor modelo salvo...")
    # ================================================================================================= #
    # Ela vai receber o modelo que teve o melhor desempenho na validação
    # ================================================================================================= #
    melhor_modelo = carregar_melhor_modelo(
        caminho_checkpoint=caminho_checkpoint,
        dispositivo=dispositivo,
        quantidade_pixels_imagem=QUANTIDADE_PIXELS_IMAGEM,
        quantidade_neuronios_ocultos=configuracao["quantidade_neuronios_ocultos"],
        quantidade_digitos=QUANTIDADE_DIGITOS,
    )

    print("\nAvaliando modelo no conjunto de teste...")
    # ================================================================================================= #
    # Aqui o código avalia o melhor_modelo usando os dados de teste
    # ================================================================================================= #
    perda_teste, acuracia_teste = avaliar_modelo(
        modelo=melhor_modelo,
        data_loader=test_loader,
        funcao_perda=funcao_perda,
        dispositivo=dispositivo,
    )

    print(f"Test loss: {perda_teste:.4f}")
    print(f"Test accuracy: {acuracia_teste:.2f}%")
    # ================================================================================================= #
    # Coletar rótulos reais e previstos. Essa função passa todas as 
    # imagens de teste pelo modelo e guarda duas listas
    # ================================================================================================= #
    y_real, y_previsto = coletar_predicoes(
        modelo=melhor_modelo,
        data_loader=test_loader,
        dispositivo=dispositivo,
    )
    # ================================================================================================= #
    # criação da matriz de confusão
    # ================================================================================================= #
    matriz_confusao = gerar_matriz_confusao(
        y_real=y_real,
        y_previsto=y_previsto,
        quantidade_digitos=QUANTIDADE_DIGITOS,
    )
    # ================================================================================================= #
    # Criar nomes das classes
    # ================================================================================================= #
    nomes_classes = [str(indice) for indice in range(QUANTIDADE_DIGITOS)]
    # ================================================================================================= #
    # Essa função gera a imagem da matriz de confusão.
    # Ela salva o arquivo dentro da pasta do experimento.
    # ================================================================================================= #
    plotar_matriz_confusao(
        matriz_confusao=matriz_confusao,
        nomes_classes=nomes_classes,
        diretorio_saida=diretorio_imagens,
    )
    # ================================================================================================= #
    # Essa função mostra algumas imagens do teste
    # ================================================================================================= #
    mostrar_predicoes_modelo(
        modelo=melhor_modelo,
        data_loader=test_loader,
        dispositivo=dispositivo,
        quantidade_imagens=16,
        diretorio_saida=diretorio_imagens,
    )
    # ================================================================================================= #
    # Preparar acurácias para comparação
    # ================================================================================================= #
    acuracia_treino_final = historico["train_accuracy"][-1] / 100
    acuracia_teste_final = acuracia_teste / 100
    # ================================================================================================= #
    # função imprime no terminal o relatório final no formato organizado.
    # ================================================================================================= #
    resumo = exibir_relatorio_avaliacao_final(
        y_real_teste=y_real,
        y_previsto_teste=y_previsto,
        acuracia_treino=acuracia_treino_final,
        acuracia_teste=acuracia_teste_final,
        quantidade_digitos=QUANTIDADE_DIGITOS,
    )
    # ================================================================================================= #
    # Montar o resultado do experimento
    # Esse dicionário guarda os principais resultados do experimento.
    # ================================================================================================= #
    resultado = {
        "experimento": nome_experimento,
        "quantidade_epocas": configuracao["quantidade_epocas"],
        "tamanho_batch": configuracao["tamanho_batch"],
        "taxa_aprendizado": configuracao["taxa_aprendizado"],
        "percentual_validacao": configuracao["percentual_validacao"],
        "quantidade_neuronios_ocultos": configuracao["quantidade_neuronios_ocultos"],
        "tipo_loss": configuracao["tipo_loss"],
        "acuracia_treino": acuracia_treino_final,
        "acuracia_teste": acuracia_teste_final,
        "diferenca_treino_teste": resumo["diferenca"],
        "melhor_val_accuracy": max(historico["val_accuracy"]) / 100,
    }

    return resultado