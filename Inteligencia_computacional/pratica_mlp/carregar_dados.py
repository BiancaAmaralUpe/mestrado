from carregar_dados_mnist.dados_mnist import carregar_dados_mnist

def executar_carregamento_dados(
    tamanho_batch: int,
    percentual_validacao: float,
    random_seed: int,
):
    """
    Executa o carregamento dos dados MNIST.

    Retorna:
        train_loader: dados de treino
        val_loader: dados de validação
        test_loader: dados de teste
    """

    print("\nCarregando dados MNIST...")

    train_loader, val_loader, test_loader = carregar_dados_mnist(
        tamanho_batch=tamanho_batch,
        percentual_validacao=percentual_validacao,
        random_seed=random_seed,
    )

    print(f"Quantidade de batches de treino: {len(train_loader)}")
    print(f"Quantidade de batches de validação: {len(val_loader)}")
    print(f"Quantidade de batches de teste: {len(test_loader)}")

    return train_loader, val_loader, test_loader