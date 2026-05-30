# pyrefly: ignore [missing-import]
# ========================================================================
#                          importações                           
# ========================================================================
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
# pyrefly: ignore [missing-import]
from tensorflow.keras.utils import to_categorical

# ========================================================================
# No primeiro teste Keras não melhorou o resultado numérico. 
# Mas ele melhorou a aprendizagem, porque agora você consegue visualizar melhor:
# camadas
# neurônios
# dropout
# loss
# val_loss
# accuracy
# val_accuracy
# épocas de treinamento
# ========================================================================
def vetorizar_textos_tfidf(X_train, X_test, max_features=3000):
    """
    Transforma os textos em vetores numéricos usando TF-IDF.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )
    #=======================================================================#
    # Aprende o vocabulário no treino e transforma o treino em matriz TF-IDF.
    # Usa o mesmo vocabulário aprendido no treino para transformar o teste                          
    #=======================================================================#
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    #=======================================================================#
    # O Keras trabalha melhor com arrays densos do tipo float32.               
    #=======================================================================#
    X_train_tfidf = X_train_tfidf.toarray().astype("float32")
    X_test_tfidf = X_test_tfidf.toarray().astype("float32")

    return X_train_tfidf, X_test_tfidf, vectorizer

def criar_modelo(input_dim, quantidade_classes):
    #=======================================================================#
    # durante o treino, 50% dos neurônios daquela camada são “desligados” 
    # temporariamente. Isso força a rede a aprender padrões mais gerais.
    # Se o TF-IDF gerou 3000 atributos, a entrada terá 3000 posições.                           
    #=======================================================================#
    modelo = Sequential([
        Input(shape=(input_dim,)),
        #=======================================================================#
        # Primeira camada oculta da MLP.
        # Dense significa que todos os neurônios dessa camada recebem todas as entradas.
        # ReLU é a função de ativação usada para aprender padrões não lineares   
        # L2 ajuda a evitar que os pesos fiquem muito grandes                      
        #=======================================================================#
        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),
        # desligou 50% dos neurônios dessa camada temporariamente
        Dropout(0.5),
        # ========================================================================#
        # Segunda camada oculta.           
        # Menor que a primeira para reduzir a complexidade do modelo.              
        #=======================================================================#
        Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),
        # Novo Dropout para reforçar a regularização.   
        Dropout(0.5),
        # Camada de Saída
        Dense(
            quantidade_classes,
            activation="softmax"
        )
    ])
    #=======================================================================#
    # compilação do modelo
    #=======================================================================#
    # adam é um otimizador que adapta a taxa de aprendizado para cada parâmetro.
    # categorical_crossentropy é usada em classificação multiclass
    # Accuracy mede o percentual de acertos.                             
    #=======================================================================#
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def treinar_modelo(modelo, X_train, y_train, epochs=50, batch_size=32):
    """
    Treina a MLP.

    epochs:
        Número máximo de épocas.
        Uma época significa uma passagem completa pelos dados de treino.
    batch_size:
        Quantidade de amostras processadas antes de atualizar os pesos.
    EarlyStopping:
        Para o treinamento quando a perda de validação não melhora.
        Isso evita que a rede continue treinando e comece a decorar o treino.
    """
    #=======================================================================#
    # numero de classes é o numero de saidas possiveis do modelo,
    # no caso do dataset de reviews de filmes, são 5 classes (1, 2, 3, 4 e 5)   
    #=======================================================================#
    quantidade_classes = modelo.output_shape[-1]

    y_train_categorico = to_categorical(
        y_train,
        num_classes=quantidade_classes
    )
#=======================================================================#
# EarlyStopping                          
#=======================================================================#
    early_stopping = EarlyStopping(
        # Monitora a perda no conjunto de validação.
        monitor="val_loss",
        # Espera 3 épocas sem melhora antes de parar
        patience=3,
        # Retorna os pesos da melhor época, não os da última
        restore_best_weights=True
    )

    print("\nIniciando treinamento da MLP com Keras")

    historico = modelo.fit(
        X_train,
        y_train_categorico,
        # # Separa 20% do treino balanceado para validação interna.
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Treinamento da MLP com Keras concluído.")

    return modelo, historico

def avaliar_modelo(modelo, X_test, y_test, label_encoder):
    print("\n" + "=" * 70)
    print("AVALIAÇÃO - MLP COM KERAS")
    print("=" * 70)

    probabilidades = modelo.predict(X_test)

    y_pred = np.argmax(probabilidades, axis=1)

    acuracia = accuracy_score(y_test, y_pred)

    y_test_texto = label_encoder.inverse_transform(y_test)
    y_pred_texto = label_encoder.inverse_transform(y_pred)

    print("\nAcurácia:")
    print(acuracia)

    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test_texto, y_pred_texto))

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_test_texto,
            y_pred_texto,
            zero_division=0
        )
    )
    return y_pred_texto
def exibir_resumo_modelo_mlp(modelo):
    """
    Exibe a arquitetura da rede neural.

    Mostra:
    - camadas
    - formato da saída de cada camada
    - quantidade de parâmetros treináveis

    """

    print("\n" + "=" * 70)
    print("ARQUITETURA DA MLP COM KERAS")
    print("=" * 70)
    modelo.summary()


def avaliar_treinamento_e_teste(
    modelo,
    X_train,
    y_train,
    X_test,
    y_test,
    label_encoder
):
    """
    Avalia o desempenho da MLP nos conjuntos de treinamento e teste.

    Essa função permite comparar:
    - desempenho no treino
    - desempenho no teste

    Se o desempenho no treino for muito superior ao teste,
    pode indicar overfitting.
    """

    print("\n" + "=" * 70)
    print("DESEMPENHO NOS CONJUNTOS DE TREINAMENTO E TESTE")
    print("=" * 70)

    # ============================================================
    # Avaliação no conjunto de treinamento
    # ============================================================

    print("\n" + "-" * 70)
    print("DESEMPENHO NO CONJUNTO DE TREINAMENTO")
    print("-" * 70)

    probabilidades_train = modelo.predict(X_train)
    y_pred_train = np.argmax(probabilidades_train, axis=1)

    acuracia_train = accuracy_score(y_train, y_pred_train)

    y_train_texto = label_encoder.inverse_transform(y_train)
    y_pred_train_texto = label_encoder.inverse_transform(y_pred_train)

    print("\nAcurácia no treinamento:")
    print(acuracia_train)

    print("\nMatriz de confusão - treinamento:")
    print(confusion_matrix(y_train_texto, y_pred_train_texto))

    print("\nRelatório de classificação - treinamento:")
    print(
        classification_report(
            y_train_texto,
            y_pred_train_texto,
            zero_division=0
        )
    )

    # ============================================================
    # Avaliação no conjunto de teste
    # ============================================================

    print("\n" + "-" * 70)
    print("DESEMPENHO NO CONJUNTO DE TESTE")
    print("-" * 70)

    probabilidades_test = modelo.predict(X_test)
    y_pred_test = np.argmax(probabilidades_test, axis=1)

    acuracia_test = accuracy_score(y_test, y_pred_test)

    y_test_texto = label_encoder.inverse_transform(y_test)
    y_pred_test_texto = label_encoder.inverse_transform(y_pred_test)

    print("\nAcurácia no teste:")
    print(acuracia_test)

    print("\nMatriz de confusão - teste:")
    print(confusion_matrix(y_test_texto, y_pred_test_texto))

    print("\nRelatório de classificação - teste:")
    print(
        classification_report(
            y_test_texto,
            y_pred_test_texto,
            zero_division=0
        )
    )

    # ============================================================
    # Comparação resumida
    # ============================================================

    diferenca = acuracia_train - acuracia_test

    print("\n" + "-" * 70)
    print("COMPARAÇÃO TREINO X TESTE")
    print("-" * 70)
    print(f"Acurácia treinamento: {acuracia_train:.4f}")
    print(f"Acurácia teste:       {acuracia_test:.4f}")
    print(f"Diferença:            {diferenca:.4f}")

    if diferenca > 0.15:
        print("\nPossível sinal de overfitting: desempenho no treino muito superior ao teste.")
    else:
        print("\nA diferença entre treino e teste não está excessivamente alta.")

    return {
        "acuracia_treinamento": acuracia_train,
        "acuracia_teste": acuracia_test,
        "diferenca": diferenca
    }
