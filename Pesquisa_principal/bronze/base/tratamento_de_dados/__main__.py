# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.transformacao_dados_xlsx_csv import transformar_base_feminicidio
# ======================================================================================= #
# para rodar o modulo de transformação dos dados de excel para .csv vai ser :
# python -m Pesquisa_principal.update_dados
# ======================================================================================= #
def main():
    transformar_base_feminicidio()

if __name__ == "__main__":
    main()