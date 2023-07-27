# entendendo o desafio e a empresa
# importação da base de dados
# usar modelo em um cenário real
import pandas as pd
from sklearn.preprocessing import LabelEncoder
tabela = pd.read_csv("clientes.csv")
display(tabela)
# print(tabela.info())
# preparação da base de dados para IA
# LabelEncoder
codificador = LabelEncoder()
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])
# print(tabela.info())
# separar as informações em dados
y = tabela["score_credito"]
# todas as colunas que se usa para fazer as previsão
# axis =1 coluna
# axis = 0 linha
x = tabela.drop(["score_credito", "id_cliente"], axis=1)
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
# criação do modelo da IA -> Score de crédito: ruim, médio, bom
# importação de IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
# escolher o melhor modelo
# calcula os acertos
from sklearn.metrics import accuracy_score
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)
# previsao_knn = modelo_knn.predict(x_teste.to_numpy)
print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsao_knn))
# processo de previsão
novos_clientes = pd.read_csv("novos_clientes.csv")
novos_clientes["profissao"] = codificador.fit_transform(novos_clientes["profissao"])
novos_clientes["mix_credito"] = codificador.fit_transform(novos_clientes["mix_credito"])
novos_clientes["comportamento_pagamento"] = codificador.fit_transform(novos_clientes["comportamento_pagamento"])
display(novos_clientes)
previsao = modelo_arvoredecisao.predict(novos_clientes)
print(previsao)
