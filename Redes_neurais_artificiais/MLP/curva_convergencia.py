# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt


def plotar_curva_convergencia(historico):
    """
    Plota as curvas de perda e acurácia durante o treinamento da MLP com Keras.

    Também identifica a melhor época com base na menor val_loss.
    """

    print("\n" + "=" * 70)
    print("CURVA DE CONVERGÊNCIA - KERAS")
    print("=" * 70)

    loss = historico.history["loss"]
    val_loss = historico.history["val_loss"]

    print(f"Loss inicial: {loss[0]:.6f}")
    print(f"Loss final: {loss[-1]:.6f}")
    print(f"Val loss inicial: {val_loss[0]:.6f}")
    print(f"Val loss final: {val_loss[-1]:.6f}")

    # Identifica a melhor época pela menor perda de validação.
    melhor_epoca = val_loss.index(min(val_loss)) + 1
    menor_val_loss = min(val_loss)

    print(f"Melhor época pela validação: {melhor_epoca}")
    print(f"Menor val_loss: {menor_val_loss:.6f}")

    if val_loss[-1] > menor_val_loss:
        print(
            "Sinal de overfitting: a val_loss aumentou após a melhor época, "
            "enquanto o treino continuou ajustando os dados."
        )
    else:
        print("Não houve aumento da val_loss após a melhor época.")

    plt.figure(figsize=(10, 6))
    plt.plot(loss, marker="o", label="Treino")
    plt.plot(val_loss, marker="o", label="Validação")
    plt.axvline(
        x=melhor_epoca - 1,
        linestyle="--",
        label=f"Melhor época: {melhor_epoca}"
    )
    plt.title("Curva de convergência - Loss - MLP com Keras")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if "accuracy" in historico.history and "val_accuracy" in historico.history:
        accuracy = historico.history["accuracy"]
        val_accuracy = historico.history["val_accuracy"]

        plt.figure(figsize=(10, 6))
        plt.plot(accuracy, marker="o", label="Treino")
        plt.plot(val_accuracy, marker="o", label="Validação")
        plt.axvline(
            x=melhor_epoca - 1,
            linestyle="--",
            label=f"Melhor época: {melhor_epoca}"
        )
        plt.title("Curva de convergência - Accuracy - MLP com Keras")
        plt.xlabel("Épocas")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()