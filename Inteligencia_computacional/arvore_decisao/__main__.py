# __main__.py
# ====================================================================================== #
# Rode assim: python -m Inteligencia_computacional.arvore_decisao
# ====================================================================================== #

from .carregar_arquivo import carregar_dataset
from .limpeza_dados import verificar_dados
from .normalizacao_dados import normalizar_dados
from .preparacao_dados import separar_treino_teste
from .tipos_arvores.arvores_restrita import treinar_arvore_restrita
from .tipos_arvores.arvores_selvagens import treinar_arvore_selvagem
from .tipos_arvores.arvores_otimizadas import treinar_arvore_otimizada
from .tipos_arvores.arvores_teoria_informacao import treinar_arvore_teoria_informacao


df = carregar_dataset("diabetes.csv")

verificar_dados(df)

# =========================================================
# Chamada que eu posso excluir depois. 
# Só queria ver se ia ser necessario normalizar os dados
# =========================================================
df = normalizar_dados(df)

# =========================================================
# Chamada para separação dos dados em treino e teste
# =========================================================
X_train, X_test, y_train, y_test = separar_treino_teste(df)

# =========================================================
# Chamada da arvore restrita limitada a 3 profundidades
# =========================================================
modelo_restrito = treinar_arvore_restrita(
    X_train,
    X_test,
    y_train,
    y_test
)

# =========================================================
# Chamada da Árvore Selvagem
# Esta configuração não define limite de profundidade. A árvore pode crescer livremente,
# =========================================================

modelo_selvagem = treinar_arvore_selvagem(
    X_train,
    X_test,
    y_train,
    y_test
)
# =========================================================
# Chamada da arvore otimizada
# 
# =========================================================
modelo_otimizado = treinar_arvore_otimizada(
    X_train,
    X_test,
    y_train,
    y_test
)

# =========================================================
# Treinamento e avaliação da quarta árvore: Teoria da Informação
# Esta configuração utiliza criterion="entropy".
# A entropia mede a impureza dos nós com base na teoria da informação.
# =========================================================
modelo_teoria_informacao = treinar_arvore_teoria_informacao(
    X_train,
    X_test,
    y_train,
    y_test
)