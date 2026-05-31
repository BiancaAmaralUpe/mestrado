
# ======================================================================================== #
# Responsável apenas pela arquitetura da MLP.
# ======================================================================================== #
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

class ModeloMLP(nn.Module):
    """
    Modelo MLP simples para classificação de dígitos manuscritos do MNIST.
    """

    def __init__(
        self,
        tamanho_entrada_pixel: int = 28 * 28,
        tamanho_neuronios_ocultos: int = 100,
        quantidade_digitos: int = 10,
    ):
        super().__init__()

        self.camada_entrada = nn.Linear(tamanho_entrada_pixel, tamanho_neuronios_ocultos)
        self.ativacao = nn.ReLU()
        self.camada_saida = nn.Linear(tamanho_neuronios_ocultos, quantidade_digitos)

    def executador(self, imagens: torch.Tensor) -> torch.Tensor:
        """
        Executa a passagem dos dados pela rede neural.
        """
        # =======================================================================
        # Transforma cada imagem 28x28 em um vetor com 784 valores.
        # =======================================================================
        imagens_vetorizadas = imagens.view(imagens.size(0), -1)

        # =======================================================================
        # Passa os pixels pela camada oculta.
        # =======================================================================
        saida_camada_oculta = self.camada_oculta(imagens_vetorizadas)

        # =======================================================================
        # Aplica a função de ativação ReLU.
        # =======================================================================
        saida_ativada = self.funcao_ativacao(saida_camada_oculta)

        # =======================================================================
        # Gera os 10 valores finais, um para cada dígito de 0 a 9.
        # =======================================================================
        logits = self.camada_classificacao(saida_ativada)

        # =======================================================================   
        # Retorna logits brutos.
        # =======================================================================
        return logits



def contar_parametros(modelo: nn.Module) -> int:
    """
    Conta a quantidade de parâmetros treináveis do modelo.
    """
    return sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)