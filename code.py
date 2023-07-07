from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Definição das características dos animais
porco1 = [0, 1, 0]
porco2 = [0, 0, 1]
porco3 = [1, 1, 0]
cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# Dados de treinamento
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]

# Criação e treinamento do modelo
model = LinearSVC()
model.fit(treino_x, treino_y)

# Dados de teste
teste_x = [[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 0]]
teste_y = [0, 0, 0, 1]

# Previsão e avaliação do modelo
previsoes = model.predict(teste_x)
taxa_de_acertos = accuracy_score(teste_y, previsoes)

print("Taxa de acerto: %.2f%%" % (taxa_de_acertos * 100))
