from .caso_and import caso_and
from .caso_or import caso_or
from .caso_xor import caso_xor


def main():
    print("Executando casos do Perceptron")
    print("====================================\n")

    caso_and()

    print("\n\n")

    caso_or()

    print("\n\n")

    caso_xor()


if __name__ == "__main__":
    main()