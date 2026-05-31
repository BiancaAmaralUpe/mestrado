# pyrefly: ignore [missing-import]
import sys

# pyrefly: ignore [missing-import]
import random

# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
from contextlib import redirect_stdout

# pyrefly: ignore [missing-import]
from .constants import (
    RANDOM_SEED,
    OUTPUT_IMAGENS_DIR,
    OUTPUT_RELATORIOS_DIR,
    CAMINHO_LOG_EXECUCAO,
)


# ======================================================================================== #
# Responsável por funções auxiliares da prática XOR.
#
# Este arquivo contém funções para:
#   - definir a semente aleatória;
#   - criar diretórios de saída;
#   - salvar todo o output do terminal em um arquivo .txt.
# ======================================================================================== #


# ======================================================================================== #
# Define a semente aleatória da execução.
#
# A seed deixa os resultados mais reprodutíveis, ou seja, ajuda a obter
# resultados iguais ou muito parecidos em diferentes execuções.
# ======================================================================================== #
def definir_semente(random_seed: int = RANDOM_SEED) -> None:
    """
    Define a semente aleatória usada na execução.

    A semente é aplicada em:
        - random;
        - numpy.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
# ======================================================================================== #
# Cria os diretórios necessários para salvar os resultados da prática.
#
# As imagens serão salvas em output-imagens.
# O log completo será salvo em output-relatorios.
# ======================================================================================== #
def criar_diretorios_necessarios() -> None:
    """
    Cria as pastas usadas pela prática XOR, caso ainda não existam.
    """

    diretorios = [
        OUTPUT_IMAGENS_DIR,
        OUTPUT_RELATORIOS_DIR,
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)


# ======================================================================================== #
# Classe usada para duplicar a saída do terminal.
#
# Tudo que for impresso com print() continua aparecendo no terminal
# e também é gravado no arquivo de log.
# ======================================================================================== #
class Tee:
    """
    Duplica a saída do terminal para um arquivo.
    Resultado:
        - aparece no terminal;
        - também é salvo no arquivo .txt.
    """

    def __init__(self, *arquivos):
        self.arquivos = arquivos

    def write(self, texto):
        """
        Escreve o texto em todos os destinos configurados.
        """

        for arquivo in self.arquivos:
            arquivo.write(texto)
            arquivo.flush()

    def flush(self):
        """
        Garante que os textos pendentes sejam gravados.
        """

        for arquivo in self.arquivos:
            arquivo.flush()


# ======================================================================================== #
# Executa uma função salvando todo o output em arquivo .txt.
#
# Essa função será usada na __main__.py para executar o fluxo completo da prática XOR.
# ======================================================================================== #
def executar_com_log(funcao_principal) -> None:
    """
    Executa uma função principal e salva todo o output gerado no terminal.

    Parâmetro:
        funcao_principal:
            função que executa a prática completa.
    """

    criar_diretorios_necessarios()

    CAMINHO_LOG_EXECUCAO.parent.mkdir(parents=True, exist_ok=True)

    with open(CAMINHO_LOG_EXECUCAO, "w", encoding="utf-8") as arquivo_log:
        saida_dupla = Tee(sys.stdout, arquivo_log)

        with redirect_stdout(saida_dupla):
            funcao_principal()

    print(f"\nLog completo salvo em: {CAMINHO_LOG_EXECUCAO}")