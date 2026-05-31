# ======================================================================================== #
# Responsável por orquestrar todos os experimentos da prática MLP com MNIST.
# ======================================================================================== #
# pyrefly: ignore [missing-import]
import sys
from contextlib import redirect_stdout

# pyrefly: ignore [missing-import]
from Inteligencia_computacional.pratica_mlp.constants import CAMINHO_RELATORIO_EXPERIMENTOS
from Inteligencia_computacional.pratica_mlp.constants import CAMINHO_LOG_EXECUCAO
from .constants import RANDOM_SEED, EXPERIMENTOS

# pyrefly: ignore [missing-import]
from .experimentos import executar_experimento

# pyrefly: ignore [missing-import]
from .utils import (
    definir_semente,
    obter_dispositivo,
    criar_diretorios_necessarios,
    exibir_informacoes_dispositivo,
    Tee,
)

# =================================================================================== #
# Para executar o módulo:
# python -m Inteligencia_computacional.pratica_mlp
# =================================================================================== #


def main():
    """
    Executa todos os experimentos da atividade MNIST com MLP.

    A lista de experimentos fica definida no arquivo constants.py.

    Cada experimento pode alterar:
        - quantidade de épocas;
        - tamanho do batch;
        - taxa de aprendizado;
        - percentual de validação;
        - tamanho da MLP;
        - função de perda.
    """

    # ======================================================================================== #
    # Configurações iniciais.
    # ======================================================================================== #
    definir_semente(random_seed=RANDOM_SEED)
    criar_diretorios_necessarios()

    dispositivo = obter_dispositivo()
    exibir_informacoes_dispositivo(dispositivo)

    resultados = []

    # ======================================================================================== #
    # Execução dos experimentos.
    #
    # Cada configuração da lista EXPERIMENTOS representa um teste diferente.
    # A função executar_experimento executa o fluxo completo:
    #   - carrega os dados;
    #   - cria a MLP;
    #   - define a loss;
    #   - treina;
    #   - avalia;
    #   - gera gráficos;
    #   - retorna os resultados principais.
    # ======================================================================================== #
    for configuracao in EXPERIMENTOS:
        resultado = executar_experimento(
            configuracao=configuracao,
            dispositivo=dispositivo,
        )

        resultados.append(resultado)

    # ======================================================================================== #
    # Resumo final dos experimentos.
    #
    # Esta tabela permite comparar o impacto das alterações solicitadas pelo professor:
    #   - N_EPOCHS;
    #   - BATCH_SIZE;
    #   - LEARNING_RATE;
    #   - VALIDATION_SPLIT;
    #   - tamanho da MLP;
    #   - função de perda.
    # ======================================================================================== #
    print("\n" + "=" * 120)
    print("RESUMO FINAL DOS EXPERIMENTOS")
    print("=" * 120)

    print(
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

    print("-" * 120)

    for resultado in resultados:
        print(
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

if __name__ == "__main__":
    criar_diretorios_necessarios()

    CAMINHO_LOG_EXECUCAO.parent.mkdir(parents=True, exist_ok=True)

    with open(CAMINHO_LOG_EXECUCAO, "w", encoding="utf-8") as arquivo_log:
        saida_dupla = Tee(sys.stdout, arquivo_log)

        with redirect_stdout(saida_dupla):
            main()

    print(f"\nLog completo salvo em: {CAMINHO_LOG_EXECUCAO}")