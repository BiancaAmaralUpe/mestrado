from pathlib import Path


# ======================================================================================== #
# Diretório base do módulo pratica_xor.
# ======================================================================================== #
BASE_DIR = Path(__file__).resolve().parent

# ======================================================================================== #
# Diretórios do projeto.
# ======================================================================================== #
OUTPUT_IMAGENS_DIR = BASE_DIR / "output-imagens"
OUTPUT_RELATORIOS_DIR = BASE_DIR / "output-relatorios"

# ======================================================================================== #
# Arquivos de saída.
# ======================================================================================== #
CAMINHO_LOG_EXECUCAO = OUTPUT_RELATORIOS_DIR / "log_execucao_completo.txt"
CAMINHO_RELATORIO_FINAL = OUTPUT_RELATORIOS_DIR / "relatorio_final_xor.txt"

# ======================================================================================== #
# Configuração geral.
# ======================================================================================== #
RANDOM_SEED = 42

# ======================================================================================== #
# Configurações do dataset XOR.
# ======================================================================================== #
NOME_COLUNA_X1 = "x1"
NOME_COLUNA_X2 = "x2"
NOME_COLUNA_ALVO = "XOR target"

QUANTIDADE_CLASSES = 2
NOMES_CLASSES = [0, 1]

# ======================================================================================== #
# Hiperparâmetros do Perceptron simples.
# ======================================================================================== #
TAXA_APRENDIZADO_PERCEPTRON = 1.0
QUANTIDADE_EPOCAS_PERCEPTRON = 20

# Ordem fixa das amostras para deixar o resultado reprodutível.
ORDEM_AMOSTRAS_PERCEPTRON = [0, 3, 1, 2]

# ======================================================================================== #
# Hiperparâmetros da MLP.
# ======================================================================================== #
CAMADAS_OCULTAS_MLP = (4,)
FUNCAO_ATIVACAO_MLP = "tanh"
SOLVER_MLP = "lbfgs"
ALPHA_MLP = 1e-5
QUANTIDADE_MAXIMA_ITERACOES_MLP = 5000
RANDOM_STATE_MLP = 0

# ======================================================================================== #
# Configurações dos gráficos.
# ======================================================================================== #
RESOLUCAO_MALHA_GRAFICO = 400
LIMITE_MINIMO_GRAFICO = -0.5
LIMITE_MAXIMO_GRAFICO = 1.5