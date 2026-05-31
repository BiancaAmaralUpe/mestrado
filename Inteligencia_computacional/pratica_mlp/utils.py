# pyrefly: ignore [missing-import]
import random
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from .constants import (
    DATA_DIR,
    RUNS_DIR,
    CHECKPOINT_DIR,
    OUTPUT_IMAGENS_DIR,
    OUTPUT_RELATORIOS_DIR,
)

# ======================================================================================== #
# Responsável por guarda funções úteis que ajudam o projeto a funcionar melhor
# ======================================================================================== #

def definir_semente(random_seed: int = 42) -> None:
    """
    Define uma seed para tentar manter os resultados reproduzíveis.

    Em redes neurais, alguns processos usam aleatoriedade, como:
        - inicialização dos pesos do modelo;
        - embaralhamento dos dados de treino;
        - algumas operações em GPU.

    Ao definir uma seed fixa, tentamos fazer com que o resultado seja
    mais parecido toda vez que o código for executado.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Essas configurações ajudam na reprodutibilidade em GPU.
    # Porém, podem deixar o treinamento um pouco mais lento.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obter_dispositivo() -> torch.device:
    """
    Verifica se existe GPU CUDA disponível.

    Se existir GPU:
        retorna cuda

    Se não existir GPU:
        retorna cpu
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criar_diretorios_necessarios() -> None:
    """
    Cria as pastas usadas pelo projeto, caso ainda não existam.
    """

    diretorios = [
        DATA_DIR,
        RUNS_DIR,
        CHECKPOINT_DIR,
        OUTPUT_IMAGENS_DIR,
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)


def exibir_informacoes_dispositivo(dispositivo: torch.device) -> None:
    """
    Exibe informações sobre o dispositivo usado no treinamento.

    Isso ajuda a saber se o código está rodando em CPU ou GPU.
    """

    print(f"Dispositivo usado: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU CUDA não disponível. O treinamento será executado na CPU.")

def criar_diretorios_experimento(nome_experimento: str):
    """
    Cria diretórios específicos para cada experimento.

    Isso evita sobrescrever imagens e checkpoints de experimentos diferentes.
    """

    diretorio_checkpoint = CHECKPOINT_DIR / nome_experimento
    diretorio_imagens = OUTPUT_IMAGENS_DIR / nome_experimento

    diretorio_checkpoint.mkdir(parents=True, exist_ok=True)
    diretorio_imagens.mkdir(parents=True, exist_ok=True)

    caminho_checkpoint = diretorio_checkpoint / "best_mlp.pth"

    return caminho_checkpoint, diretorio_imagens

def criar_diretorios_necessarios() -> None:
    """
    Cria as pastas usadas pelo projeto, caso ainda não existam.

    As pastas são criadas dentro do módulo pratica_mlp:
        - data
        - runs
        - checkpoints
        - output-imagens
        - output-relatorios
    """

    diretorios = [
        DATA_DIR,
        RUNS_DIR,
        CHECKPOINT_DIR,
        OUTPUT_IMAGENS_DIR,
        OUTPUT_RELATORIOS_DIR,
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)

def salvar_resumo_experimentos_txt(
    resultados: list,
    caminho_relatorio,
) -> None:
    """
    Salva em arquivo .txt o resumo final dos experimentos executados.

    O arquivo gerado permite consultar os resultados depois,
    sem depender apenas do log exibido no terminal.
    """

    linhas = []

    linhas.append("=" * 120)
    linhas.append("RESUMO FINAL DOS EXPERIMENTOS")
    linhas.append("=" * 120)
    linhas.append("")

    cabecalho = (
        f"{'Experimento':<22} "
        f"{'Epochs':<8} "
        f"{'Batch':<8} "
        f"{'LR':<12} "
        f"{'ValSplit':<10} "
        f"{'Hidden':<8} "
        f"{'Loss':<16} "
        f"{'ValAcc':<10} "
        f"{'TestAcc':<10} "
        f"{'Gap':<10}"
    )

    linhas.append(cabecalho)
    linhas.append("-" * 120)

    for resultado in resultados:
        linha = (
            f"{resultado['experimento']:<22} "
            f"{resultado['quantidade_epocas']:<8} "
            f"{resultado['tamanho_batch']:<8} "
            f"{resultado['taxa_aprendizado']:<12} "
            f"{resultado['percentual_validacao']:<10} "
            f"{resultado['quantidade_neuronios_ocultos']:<8} "
            f"{resultado['tipo_loss']:<16} "
            f"{resultado['melhor_val_accuracy']:<10.4f} "
            f"{resultado['acuracia_teste']:<10.4f} "
            f"{resultado['diferenca_treino_teste']:<10.4f}"
        )

        linhas.append(linha)

    linhas.append("")
    linhas.append("=" * 120)
    linhas.append("LEGENDA")
    linhas.append("=" * 120)
    linhas.append("Epochs   = quantidade de épocas usadas no treinamento.")
    linhas.append("Batch    = tamanho do batch.")
    linhas.append("LR       = learning rate, ou taxa de aprendizado.")
    linhas.append("ValSplit = percentual do treino separado para validação.")
    linhas.append("Hidden   = quantidade de neurônios na camada oculta da MLP.")
    linhas.append("Loss     = função de perda usada no experimento.")
    linhas.append("ValAcc   = melhor acurácia de validação.")
    linhas.append("TestAcc  = acurácia no conjunto de teste.")
    linhas.append("Gap      = diferença entre acurácia de treino e teste.")
    linhas.append("")

    caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_relatorio, "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(linhas))

    print(f"\nRelatório salvo em: {caminho_relatorio}")

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