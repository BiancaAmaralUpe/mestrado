# ================================================================================
# Para rodar esse modulo, é preciso ter instalado:
# python -m pip install pandas matplotlib scikit-learn tensorflow
# para rodar o modulo:  python -m Redes_neurais_artificiais.MLP
# ================================================================================

from pathlib import Path

from .curva_convergencia import plotar_curva_convergencia
from .carregar_dataset import carregar_dataset
from .limpeza_dados import limpar_dados
from .normalizar_dados import separar_treino_teste
from .balancear_dados import balancear_treino_oversampling
from .modelo_mlp import (
    vetorizar_textos_tfidf,
    criar_modelo,
    treinar_modelo,
    avaliar_modelo,
    avaliar_treinamento_e_teste,
    exibir_resumo_modelo_mlp
)


def main():
    print("=" * 70)
    print("Classificação de Reviews de Animes com MLP - Keras")
    print("=" * 70)

    pasta_atual = Path(__file__).parent
    caminho_dataset = pasta_atual / "dados" / "anime_reviews.csv"

    df = carregar_dataset(caminho_dataset)

    df_limpo = limpar_dados(df)

    X_train, X_test, y_train, y_test, label_encoder = separar_treino_teste(df_limpo)

    # Random Oversampling somente no conjunto de treino.
    X_train_balanceado, y_train_balanceado = balancear_treino_oversampling(
        X_train,
        y_train,
        label_encoder
    )

    # TF-IDF:
    # fit_transform no treino balanceado
    # transform no teste original
    X_train_tfidf, X_test_tfidf, vectorizer = vetorizar_textos_tfidf(
        X_train_balanceado,
        X_test
    )

    input_dim = X_train_tfidf.shape[1]
    quantidade_classes = len(label_encoder.classes_)

    modelo = criar_modelo(
        input_dim=input_dim,
        quantidade_classes=quantidade_classes
    )

    exibir_resumo_modelo_mlp(modelo)

    modelo_treinado, historico = treinar_modelo(
        modelo,
        X_train_tfidf,
        y_train_balanceado,
        epochs=50,
        batch_size=32
    )

    avaliar_modelo(
        modelo_treinado,
        X_test_tfidf,
        y_test,
        label_encoder
    )

    avaliar_treinamento_e_teste(
        modelo_treinado,
        X_train_tfidf,
        y_train_balanceado,
        X_test_tfidf,
        y_test,
        label_encoder
    )

    plotar_curva_convergencia(historico)


if __name__ == "__main__":
    main()