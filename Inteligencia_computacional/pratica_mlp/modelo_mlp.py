
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
        quantidade_pixels_imagem: int = 28 * 28,
        quantidade_neuronios_ocultos: int = 100,
        quantidade_digitos: int = 10,
    ):
        super().__init__()

        self.camada_oculta = nn.Linear(
            quantidade_pixels_imagem,
            quantidade_neuronios_ocultos,
        )

        self.funcao_ativacao = nn.ReLU()

        self.camada_classificacao = nn.Linear(
            quantidade_neuronios_ocultos,
            quantidade_digitos,
        )

    def forward(self, imagens: torch.Tensor) -> torch.Tensor:
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
        
def contar_parametros_treinaveis(modelo: nn.Module) -> int:
    """
    Conta a quantidade de parâmetros treináveis do modelo.
    """
    return sum(
        parametro.numel()
        for parametro in modelo.parameters()
        if parametro.requires_grad
    )