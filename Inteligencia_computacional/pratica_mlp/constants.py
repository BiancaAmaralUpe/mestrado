from pathlib import Path


# ======================================================================================== #
# Diretório base do módulo pratica_mlp.
# ======================================================================================== #
BASE_DIR = Path(__file__).resolve().parent

# ======================================================================================== #
# Diretórios do projeto.
# ======================================================================================== #
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_IMAGENS_DIR = BASE_DIR / "output-imagens"
OUTPUT_RELATORIOS_DIR = BASE_DIR / "output-relatorios"

# ======================================================================================== #
# Arquivos de saída.
# ======================================================================================== #
CAMINHO_RELATORIO_EXPERIMENTOS = OUTPUT_RELATORIOS_DIR / "resumo_experimentos.txt"
CAMINHO_LOG_EXECUCAO = OUTPUT_RELATORIOS_DIR / "log_execucao_completo.txt"

# ======================================================================================== #
# Hiperparâmetros principais - baseline.
# ======================================================================================== #
RANDOM_SEED = 42

QUANTIDADE_PIXELS_IMAGEM = 28 * 28
QUANTIDADE_NEURONIOS_OCULTOS = 100
QUANTIDADE_DIGITOS = 10

QUANTIDADE_EPOCAS = 20
TAMANHO_BATCH = 4096
TAXA_APRENDIZADO = 0.001
PERCENTUAL_VALIDACAO = 0.10

TIPO_LOSS = "cross_entropy"

CAMINHO_CHECKPOINT = CHECKPOINT_DIR / "best_mlp.pth"

# ======================================================================================== #
# Experimentoos
# ======================================================================================== #
EXPERIMENTOS = [
    {
        "nome": "baseline",
        "quantidade_epocas": 20,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "epocas_10",
        "quantidade_epocas": 10,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "batch_256",
        "quantidade_epocas": 20,
        "tamanho_batch": 256,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "lr_0001",
        "quantidade_epocas": 20,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.0001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "validacao_20",
        "quantidade_epocas": 20,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.20,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "mlp_maior",
        "quantidade_epocas": 20,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 256,
        "tipo_loss": "cross_entropy",
    },
    {
        "nome": "loss_multimargin",
        "quantidade_epocas": 20,
        "tamanho_batch": 4096,
        "taxa_aprendizado": 0.001,
        "percentual_validacao": 0.10,
        "quantidade_neuronios_ocultos": 100,
        "tipo_loss": "multi_margin",
    },
]