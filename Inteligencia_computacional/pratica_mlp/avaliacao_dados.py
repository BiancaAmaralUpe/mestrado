# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# pyrefly: ignore [missing-import]
from carregar_dados_mnist.modelo_mlp import ModeloMLP

# ===================================================================================================== #
# accuracy_score -> acurácia geral
# balanced_accuracy_score ->  acurácia balanceada
# classification_report -> relatório completo por dígito
# confusion_matrix -> matriz de confusão
# precision_recall_fscore_support -> precision, recall e f1-score
# ===================================================================================================== #

# ===================================================================================================== #
# Responsável por carregar o melhor modelo, avaliar no teste, coletar predições e gerar métricas finais.
# ===================================================================================================== #

def carregar_melhor_modelo(
    caminho_checkpoint: str,
    dispositivo: torch.device,
    quantidade_pixels_imagem: int = 28 * 28,
    quantidade_neuronios_ocultos: int = 100,
    quantidade_digitos: int = 10,
) -> ModeloMLP:
    """
    Carrega o melhor modelo salvo durante o treinamento.
    """
    # ================================================================================================= #
    # Carrega em checkpoint, o melhor modelo treinado
    # ================================================================================================= #
    checkpoint = torch.load(caminho_checkpoint, map_location=dispositivo)

    # ================================================================================================= #
    # Instancia o modelo.
    # Criação de uma mlp nova com a mesma estrutura utilizada no treino
    # ================================================================================================= #
    modelo = ModeloMLP(
        quantidade_pixels_imagem=quantidade_pixels_imagem,
        quantidade_neuronios_ocultos=quantidade_neuronios_ocultos,
        quantidade_digitos=quantidade_digitos,
    ).to(dispositivo)
    # ================================================================================================= #
    # insere os pesos do modelo, e o modelo em .aval entra em avaliação
    # ================================================================================================= #
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    print(f"Checkpoint carregado de: {caminho_checkpoint}")

    if "best_val_accuracy" in checkpoint:
        print(f"Melhor acurácia de validação: {checkpoint['best_val_accuracy']:.2f}%")

    if "epoch" in checkpoint:
        print(f"Época do melhor checkpoint: {checkpoint['epoch'] + 1}")

    return modelo

# ====================================================================================================== #
# Calcula a função da loss media , acuracia, e AVALIA validacao e teste
# @torch.no_grad() => Não calcule gradientes.
#                     Não prepare o modelo para aprender.
#                     Só faça previsão.
# ====================================================================================================== #
@torch.no_grad()

def avaliar_modelo(
    modelo: nn.Module,
    data_loader,
    funcao_perda,
    dispositivo: torch.device,
):
    """
    Avalia o modelo em um conjunto de dados.

    Essa função pode ser usada para validação ou teste.

    Durante a avaliação:
        - os pesos do modelo não são atualizados;
        - o modelo apenas faz previsões;
        - são calculadas loss média e acurácia.

    Parâmetros:
        modelo:
            Modelo treinado.

        data_loader:
            DataLoader de validação ou teste.

        funcao_perda:
            Função de perda usada para calcular o erro.

        dispositivo:
            CPU ou GPU.

    Retorno:
        perda_media:
            Loss média no conjunto avaliado.

        acuracia:
            Percentual de acertos do modelo.
    """

    modelo.eval()

    perda_total = 0.0
    quantidade_acertos = 0
    quantidade_amostras = 0

    for imagens, rotulos_reais in data_loader:
        imagens = imagens.to(dispositivo)
        rotulos_reais = rotulos_reais.to(dispositivo)

        logits = modelo(imagens)
        perda = funcao_perda(logits, rotulos_reais)

        tamanho_lote = rotulos_reais.size(0)

        perda_total += perda.item() * tamanho_lote

        rotulos_previstos = torch.argmax(logits, dim=1)

        quantidade_acertos += (rotulos_previstos == rotulos_reais).sum().item()
        quantidade_amostras += tamanho_lote

    perda_media = perda_total / quantidade_amostras
    acuracia = 100.0 * quantidade_acertos / quantidade_amostras

    return perda_media, acuracia



# ====================================================================================================== #
# Essa função serve para guardar todas as respostas.
# ====================================================================================================== #
@torch.no_grad()
def coletar_predicoes(
    modelo: nn.Module,
    data_loader,
    dispositivo: torch.device,
):
    """
    Coleta os rótulos reais e os rótulos previstos pelo modelo.

    Essa função percorre todo o conjunto de teste e guarda:

        y_real:
            Classes verdadeiras das imagens.

        y_previsto:
            Classes previstas pela MLP.

    Esses vetores são usados para gerar:
        - matriz de confusão;
        - classification report;
        - métricas finais.
    """

    modelo.eval()

    y_real = []
    y_previsto = []

    for imagens, rotulos_reais in data_loader:
        imagens = imagens.to(dispositivo)

        logits = modelo(imagens)
        rotulos_previstos = torch.argmax(logits, dim=1).cpu().numpy()

        y_real.extend(rotulos_reais.numpy())
        y_previsto.extend(rotulos_previstos)

    return np.array(y_real), np.array(y_previsto)
# ====================================================================================================== #
# Essa função cria a matriz de confusão.
# ====================================================================================================== #
def gerar_matriz_confusao(
    y_real,
    y_previsto,
    quantidade_digitos: int = 10,
):
    """
    Gera a matriz de confusão.

    Linhas:
        representam os rótulos reais.

    Colunas:
        representam os rótulos previstos pelo modelo.
    """

    return confusion_matrix(
        y_real,
        y_previsto,
        labels=list(range(quantidade_digitos)),
    )

# ====================================================================================================== #
# Essa função mostra o resultado final do modelo , ela calcula várias métricas.
# ====================================================================================================== #
def exibir_metricas_finais(
    y_real,
    y_previsto,
    quantidade_digitos: int = 10,
):
    """
    Exibe as métricas finais do modelo no conjunto de teste.

    Métricas exibidas:
        - Accuracy;
        - Balanced accuracy;
        - Macro precision;
        - Macro recall;
        - Macro F1-score;
        - Weighted precision;
        - Weighted recall;
        - Weighted F1-score;
        - Classification report por dígito.
    """

    acuracia = accuracy_score(y_real, y_previsto)
    acuracia_balanceada = balanced_accuracy_score(y_real, y_previsto)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_real,
        y_previsto,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_real,
        y_previsto,
        average="weighted",
        zero_division=0,
    )

    print("\nMétricas finais no conjunto de teste")
    print("-" * 80)
    print(f"Accuracy:            {acuracia:.4f}")
    print(f"Balanced accuracy:   {acuracia_balanceada:.4f}")
    print(f"Macro precision:     {macro_precision:.4f}")
    print(f"Macro recall:        {macro_recall:.4f}")
    print(f"Macro F1-score:      {macro_f1:.4f}")
    print(f"Weighted precision:  {weighted_precision:.4f}")
    print(f"Weighted recall:     {weighted_recall:.4f}")
    print(f"Weighted F1-score:   {weighted_f1:.4f}")

    nomes_classes = [f"digit_{indice}" for indice in range(quantidade_digitos)]

    print("\nClassification report")
    print("-" * 80)
    print(
        classification_report(
            y_real,
            y_previsto,
            target_names=nomes_classes,
            digits=4,
            zero_division=0,
        )
    )