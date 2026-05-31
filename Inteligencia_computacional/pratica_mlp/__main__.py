# ======================================================================================== #
# Responsável por orquestrar tudo
# ======================================================================================== #

# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from carregar_dados_mnist.dados_mnist import carregar_dados_mnist
from carregar_dados_mnist.modelo_mlp import ModeloMLP
from carregar_dados_mnist.modelo_mlp import contar_parametros_treinaveis
# pyrefly: ignore [missing-import]
from carregar_dados_mnist.treinamento import treinar_modelo
from carregar_dados_mnist.avaliacao_dados import carregar_melhor_modelo
from carregar_dados_mnist.avaliacao_dados import avaliar_modelo
from carregar_dados_mnist.avaliacao_dados import coletar_predicoes
from carregar_dados_mnist.avaliacao_dados import gerar_matriz_confusao
from carregar_dados_mnist.avaliacao_dados import exibir_metricas_finais
# pyrefly: ignore [missing-import]
from carregar_dados_mnist.geracao_graficos import mostrar_exemplos_mnist
from carregar_dados_mnist.geracao_graficos import plotar_historico_treinamento
from carregar_dados_mnist.geracao_graficos import plotar_matriz_confusao
from carregar_dados_mnist.geracao_graficos import mostrar_predicoes_modelo
from carregar_dados_mnist.geracao_graficos import
# pyrefly: ignore [missing-import]
from carregar_dados_mnist.utils import definir_semente
from carregar_dados_mnist.utils import obter_dispositivo
from carregar_dados_mnist.utils import criar_diretorios_necessarios
from carregar_dados_mnist.utils import exibir_informacoes_dispositivo


