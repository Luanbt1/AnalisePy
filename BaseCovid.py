# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 1. Carregar o conjunto de dados
# Substitua 'caminho_para_o_arquivo.csv' pelo caminho do seu arquivo de dados
df = pd.read_csv('lbp-test.csv')

# Verificar a estrutura do dataset
print(df.head())

# Separar a variável alvo (substitua 'class' pelo nome correto da coluna)
X = df.drop(columns=['class'])
y = df['class']

# 2. Tratar o desbalanceamento - escolha uma das opções abaixo

# Opção 1: Usar SMOTE ajustando o número de vizinhos
# smote = SMOTE(random_state=42, k_neighbors=2)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# Opção 2: Usar RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 4. Treinar modelos individuais
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 5. Criar o Ensemble com Voting Classifier
voting_ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='soft'  # 'soft' usa as probabilidades de cada modelo
)

# Treinar o ensemble
voting_ensemble.fit(X_train, y_train)

# 6. Fazer previsões e avaliar o desempenho
y_pred = voting_ensemble.predict(X_test)

# Exibir métricas de desempenho
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
