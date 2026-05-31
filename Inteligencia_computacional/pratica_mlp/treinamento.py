# pyrefly: ignore [missing-import]
import time

# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
import torch.nn as nn

# ======================================================================================== #
# Responsável por treinar o modelo, validar a cada época e salvar o melhor checkpoint.
# ======================================================================================== #


# ======================================================================================== #
# Essa função treina o modelo uma vez em todo o conjunto de treino.
# ======================================================================================== #
def treinar_uma_epoca(
    modelo: nn.Module,
    train_loader,
    funcao_perda,
    otimizador,
    dispositivo: torch.device,
    epoca: int,
):
    """
    Treina o modelo por uma época.

    Uma época significa que o modelo passou uma vez por todo o conjunto de treino.

    Durante o treino:
        - o modelo faz previsões;
        - calcula o erro;
        - calcula os gradientes;
        - atualiza os pesos;
        - calcula loss e acurácia da época.
    """

    modelo.train()

    perda_total = 0.0
    quantidade_acertos = 0
    quantidade_amostras = 0

    for imagens, rotulos_reais in train_loader:
        imagens = imagens.to(dispositivo)
        rotulos_reais = rotulos_reais.to(dispositivo)

        # ========================================================================================
        # Forward pass:
        # O modelo recebe as imagens e gera os logits.
        # ========================================================================================
        logits = modelo(imagens)

        # ========================================================================================
        # Calcula a perda comparando os logits com os rótulos reais.
        # ========================================================================================
        perda = funcao_perda(logits, rotulos_reais)

        # ========================================================================================
        # Zera os gradientes antigos.
        # ========================================================================================
        otimizador.zero_grad()

        # ========================================================================================
        # Backward pass:
        # Calcula os gradientes da perda em relação aos parâmetros do modelo.
        # ========================================================================================
        perda.backward()

        # ========================================================================================
        # Atualiza os pesos do modelo.
        # ========================================================================================
        otimizador.step()

        # ========================================================================================
        # Calcula métricas do lote atual.
        # ========================================================================================
        tamanho_lote = rotulos_reais.size(0)

        perda_total += perda.item() * tamanho_lote

        rotulos_previstos = torch.argmax(logits, dim=1)

        quantidade_acertos += (rotulos_previstos == rotulos_reais).sum().item()
        quantidade_amostras += tamanho_lote

    perda_media = perda_total / quantidade_amostras
    acuracia = 100.0 * quantidade_acertos / quantidade_amostras

    return perda_media, acuracia

# ======================================================================================== #
# Essa função é parecida com o treino, mas com uma diferença importante: Aqui o modelo não aprende.
# Assim como no metodo avaliação_dados.py , o @torch.no_grad() continua tendo a função de :
# fazer as previsões e calcular o 
#       *loss
#       *acuracia
# ======================================================================================== #
@torch.no_grad()
def avaliar_uma_epoca(
    modelo: nn.Module,
    data_loader,
    funcao_perda,
    dispositivo: torch.device,
):
    """
    Avalia o modelo em um conjunto de dados.

    Essa função pode ser usada para:
        - validação durante o treinamento;
        - teste após o treinamento.
    Durante a avaliação:
        - os pesos não são atualizados;
        - não há cálculo de gradientes;
        - o modelo apenas faz previsões.
    """

    modelo.eval()

    perda_total = 0.0
    quantidade_acertos = 0
    quantidade_amostras = 0

    for imagens, rotulos_reais in data_loader:
        imagens = imagens.to(dispositivo)
        rotulos_reais = rotulos_reais.to(dispositivo)

        # ========================================================================================
        # O modelo faz a previsão.
        # ========================================================================================
        logits = modelo(imagens)

        # ========================================================================================
        # Calcula a perda.
        # ========================================================================================
        perda = funcao_perda(logits, rotulos_reais)

        tamanho_lote = rotulos_reais.size(0)

        perda_total += perda.item() * tamanho_lote

        rotulos_previstos = torch.argmax(logits, dim=1)

        quantidade_acertos += (rotulos_previstos == rotulos_reais).sum().item()
        quantidade_amostras += tamanho_lote

    perda_media = perda_total / quantidade_amostras
    acuracia = 100.0 * quantidade_acertos / quantidade_amostras

    return perda_media, acuracia

# ======================================================================================== #
# Essa é a função que controla o treinamento completo.
# ======================================================================================== #
def treinar_modelo(
    modelo: nn.Module,
    train_loader,
    val_loader,
    funcao_perda,
    otimizador,
    quantidade_epocas: int,
    caminho_checkpoint: str,
    dispositivo: torch.device,
    quantidade_pixels_imagem: int = 28 * 28,
    quantidade_neuronios_ocultos: int = 100,
    quantidade_digitos: int = 10,
    taxa_aprendizado: float = 0.001,
    tamanho_batch: int = 4096,
):
    """
    Treina o modelo por várias épocas e salva o melhor checkpoint.

    O melhor checkpoint é escolhido com base na maior acurácia de validação.

    Parâmetros:
        modelo:
            Modelo MLP que será treinado.

        train_loader:
            DataLoader do conjunto de treino.

        val_loader:
            DataLoader do conjunto de validação.

        funcao_perda:
            Função usada para medir o erro do modelo.

        otimizador:
            Algoritmo responsável por atualizar os pesos do modelo.

        quantidade_epocas:
            Número de vezes que o modelo passará pelo conjunto de treino.

        caminho_checkpoint:
            Caminho onde o melhor modelo será salvo.

        dispositivo:
            CPU ou GPU.

    Retorno:
        historico:
            Dicionário com as losses e acurácias de treino e validação.
    """

    historico = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    melhor_acuracia_validacao = 0.0
    tempo_inicio = time.time()

    for epoca in range(quantidade_epocas):
        perda_treino, acuracia_treino = treinar_uma_epoca(
            modelo=modelo,
            train_loader=train_loader,
            funcao_perda=funcao_perda,
            otimizador=otimizador,
            dispositivo=dispositivo,
            epoca=epoca,
        )

        perda_validacao, acuracia_validacao = avaliar_uma_epoca(
            modelo=modelo,
            data_loader=val_loader,
            funcao_perda=funcao_perda,
            dispositivo=dispositivo,
        )

        historico["train_loss"].append(perda_treino)
        historico["train_accuracy"].append(acuracia_treino)
        historico["val_loss"].append(perda_validacao)
        historico["val_accuracy"].append(acuracia_validacao)

        # ========================================================================================
        # Salva o melhor modelo com base na acurácia de validação.
        # ========================================================================================
        if acuracia_validacao > melhor_acuracia_validacao:
            melhor_acuracia_validacao = acuracia_validacao

            torch.save(
                {
                    "epoch": epoca,
                    "model_state_dict": modelo.state_dict(),
                    "optimizer_state_dict": otimizador.state_dict(),
                    "best_val_accuracy": melhor_acuracia_validacao,
                    "hyperparameters": {
                        "quantidade_pixels_imagem": quantidade_pixels_imagem,
                        "quantidade_neuronios_ocultos": quantidade_neuronios_ocultos,
                        "quantidade_digitos": quantidade_digitos,
                        "taxa_aprendizado": taxa_aprendizado,
                        "tamanho_batch": tamanho_batch,
                    },
                },
                caminho_checkpoint,
            )

        print(
            f"Época [{epoca + 1:02d}/{quantidade_epocas}] "
            f"Train loss: {perda_treino:.4f} | "
            f"Train acc: {acuracia_treino:.2f}% | "
            f"Val loss: {perda_validacao:.4f} | "
            f"Val acc: {acuracia_validacao:.2f}%"
        )

    tempo_total = time.time() - tempo_inicio

    print(f"\nTreinamento finalizado em {tempo_total:.2f} segundos")
    print(f"Melhor acurácia de validação: {melhor_acuracia_validacao:.2f}%")

    return historico