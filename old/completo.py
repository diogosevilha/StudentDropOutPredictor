# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Bibliotecas básicas
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento e Redução de Dimensionalidade
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Ferramentas de imputação e seleção de atributos
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest

# Modelos de classificação
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Encadeamento de transformações e estimadores
from sklearn.pipeline import Pipeline

# Validação e métricas
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import roc_curve, auc

# Testes estatísticos
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

# Ajustes pandas
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# -----------------------------------------------------------------------------
# PARTE 1: PRÉ-PROCESSAMENTO (baseado em pre_processamento.py)
# -----------------------------------------------------------------------------

# 1. Carrega a base de dados ".csv"
df = pd.read_csv("microdados_eficiencia_academica_2023.csv", sep=';', decimal=',')

# Cria uma cópia de trabalho
data = df.copy()

# Exclusão de colunas referentes a processo seletivo
data.drop(columns=[
    'Vagas Regulares AC','Vagas Regulares l1','Vagas Regulares l10','Vagas Regulares l13',
    'Vagas Regulares l14','Vagas Regulares l2','Vagas Regulares l5','Vagas Regulares l6','Vagas Regulares l9',
    'Vagas Extraordinárias AC','Vagas Extraordinárias l1','Vagas Extraordinárias l10','Vagas Extraordinárias l13',
    'Vagas Extraordinárias l14','Vagas Extraordinárias l2','Vagas Extraordinárias l5','Vagas Extraordinárias l6','Vagas Extraordinárias l9'
], axis=1, inplace=True)

# Exclusão de colunas de identificação
data.drop(columns=[
    'Co Inst','Cod Unidade','Código da Matricula','Código da Unidade de Ensino - SISTEC',
    'Código do Ciclo Matricula','Código do Município com DV','Número de registros'
], axis=1, inplace=True)

# Exclusão de colunas redundantes
data.drop(columns=['Situação de Matrícula'], axis=1, inplace=True)

# Exclusão de cursos FIC
data = data[data['Tipo de Curso'] != 'Qualificação Profissional (FIC)']

# Exclusão de colunas não utilizadas ou com erro
data.drop(columns=[
    'Município','Região','Matrícula Atendida','Ano','Idade','Cor / Raça',
    'Sexo','Instituição'
], axis=1, inplace=True)

# Ajuste do rótulo
data['Categoria da Situação'] = data['Categoria da Situação'].replace({'Concluídos':'Concluintes'})

# Criação de nova feature: duracao_curso
data['duracao_curso'] = data['Carga Horaria'] / data['Carga Horaria Mínima']

# Converte datas
data['Data de Fim Previsto do Ciclo'] = pd.to_datetime(data['Data de Fim Previsto do Ciclo'], dayfirst=True)
data['Data de Inicio do Ciclo']       = pd.to_datetime(data['Data de Inicio do Ciclo'], dayfirst=True)
data['Data de Ocorrencia da Matricula'] = pd.to_datetime(data['Data de Ocorrencia da Matricula'], dayfirst=True)
data['Mês De Ocorrência da Situação']   = pd.to_datetime(data['Mês De Ocorrência da Situação'], dayfirst=True)

# Criação de nova feature: tempo_curso
data['tempo_curso'] = (
    data['Mês De Ocorrência da Situação'] - data['Data de Ocorrencia da Matricula']
) / (
    data['Data de Fim Previsto do Ciclo'] - data['Data de Ocorrencia da Matricula']
)

# Exclusão de colunas relacionadas às datas já utilizadas
data.drop(columns=[
    'Carga Horaria','Carga Horaria Mínima',
    'Data de Fim Previsto do Ciclo','Data de Inicio do Ciclo',
    'Mês De Ocorrência da Situação','Data de Ocorrencia da Matricula'
], axis=1, inplace=True)

# Renomeia colunas
data.rename(columns={
    'Categoria da Situação':'situacao',
    'Eixo Tecnológico':'eixo_tec',
    'Faixa Etária':'faixa_etaria',
    'Fator Esforço Curso':'esforco',
    'Fonte de Financiamento':'financiamento',
    'Instituição':'instituicao',
    'Modalidade de Ensino':'modalidade',
    'Nome de Curso':'curso',
    'Renda Familiar':'renda_familiar',
    'Subeixo Tecnológico':'subeixo_tec',
    'Tipo de Curso':'tipo_curso',
    'Tipo de Oferta':'tipo_oferta',
    'Turno':'turno',
    'UF':'uf',
    'Unidade de Ensino':'unidade'
}, inplace=True)

# Carrega dados de indicadores das unidades
df_unidades = pd.read_excel("Indicadores das Unidades.xlsx")

# Faz o merge (left join)
data_1 = pd.merge(data, df_unidades, on='unidade', how='left')

# Ajuste de colunas após o merge
# (caso exista 'instituicao' duplicada, pois você já removeu acima ou usou no merge)
if 'instituicao' in data_1.columns:
    data_1.drop(columns=['instituicao'], inplace=True)

data_1.rename(columns={'Município_y':'Município'}, inplace=True)

# Mapeia o alvo (situacao) em 0 (Concluintes / Em Curso) e 1 (Evadidos)
map_situacao_matricula = {
    'Concluintes': 0,
    'Em Curso': 0,
    'Evadidos': 1
}
data_1['situacao'] = data_1['situacao'].map(map_situacao_matricula)

# Mapeia faixa etária em valores ordinais (exemplo de mapeamento)
map_faixa_etaria = {
    'Menor de 14 anos':1, '15 a 19 anos':2, '20 a 24 anos':3, '25 a 29 anos':4,
    '30 a 34 anos':5, '35 a 39 anos':6, '40 a 44 anos':7, '45 a 49 anos':8,
    '50 a 54 anos':9, 'Maior de 60 anos':10, '55 a 59 anos':11, 'S/I':0
}
data_1['faixa_etaria'] = data_1['faixa_etaria'].map(map_faixa_etaria)

# LabelEncoder para as colunas do tipo objeto
lb = LabelEncoder()
objList = data_1.select_dtypes(include="object").columns

for feat in objList:
    data_1[feat] = lb.fit_transform(data_1[feat].astype(str))

# Data_1 pré-processado e pronto para análise

print("Dados após o pré-processamento:")
print(data_1.head())
print(data_1.info())

# Opcional: Vamos usar um sample de 5000 linhas para teste, como no exemplo
data_2 = data_1.sample(5000, random_state=42)

# Separa as features (X) e o alvo (y)
X = data_2.drop('situacao', axis=1)
y = data_2['situacao']

# Se preferir dividir em treino/teste para conferir
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=123)


# -----------------------------------------------------------------------------
# PARTE 2: FLUXO DE MODELAGEM (baseado em main_v3.py, adaptado)
# -----------------------------------------------------------------------------

# Define um seed para reprodutibilidade
seed = 42

# Define o scorer como acurácia (poderia usar F1, precision, recall...)
scorer = make_scorer(accuracy_score)     # acurácia
# scorer = make_scorer(precision_score)  # precisão (quantos dos positivos previstos são realmente positivos)
# scorer = make_scorer(recall_score)     #  sensibilidade (recall) (quantos dos positivos reais foram identificados)
# scorer = make_scorer(f1_score)         # equilíbrio entre precisão e recall (média harmônica entre os dois)

# Estratégia de validação cruzada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)   # Para avaliação final
gscv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)  # Para grid search

# Dicionário de algoritmos e parâmetros (GridSearchCV)
jobs = -1  # usar todos os núcleos

algorithms = {
    'kNN': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler(feature_range=(0, 1))),
            ('selector', VarianceThreshold()),
            ('knn', KNeighborsClassifier())
        ]),
        param_grid={
            'selector__threshold': [0, 0.01, 0.02, 0.03],
            'knn__n_neighbors': [1, 3, 5],
            'knn__p': [1, 2],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'tree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('tree', DecisionTreeClassifier(random_state=seed))
        ]),
        param_grid={
            'tree__max_depth': [5, 10, 20],
            'tree__criterion': ['entropy', 'gini'],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'bigtree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('tree', DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=seed))
        ]),
        param_grid={
            'tree__criterion': ['entropy', 'gini'],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'nb': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('selector', SelectKBest()),
            ('nb', GaussianNB())
        ]),
        param_grid={
            'selector__k': [3, 5, 10],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'svmlinear': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('svm', SVC(kernel='linear', random_state=seed))
        ]),
        param_grid={
            'pca__n_components': [2, 5, 10],
            'svm__C': [1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
    'svmrbf': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=seed)),
            ('svm', SVC(kernel='rbf', random_state=seed))
        ]),
        param_grid={
            'pca__n_components': [2, 5, 10],
            'svm__C': [1.0, 2.0],
            'svm__gamma': [0.1, 1.0, 2.0],
        },
        scoring=scorer,
        cv=gscv,
        n_jobs=jobs
    ),
}


# -----------------------------------------------------------------------------
# Treinamento (Validação Cruzada)
# -----------------------------------------------------------------------------
result = {}
for alg, clf in algorithms.items():
    print(f"\nTreinando e avaliando: {alg}")
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scorer, n_jobs=jobs)
    result[alg] = scores
    print(f"Acurácias (cross-val): {scores}")
    print(f"Média: {scores.mean():.4f} | Desvio: {scores.std():.4f}")

# Converte para DataFrame
result_df = pd.DataFrame.from_dict(result)
print("\nResultados de cross_val_score:")
print(result_df)

# -----------------------------------------------------------------------------
# Comparação Estatística
# -----------------------------------------------------------------------------

# Boxplot dos resultados
plt.boxplot([scores for alg, scores in result_df.items()])
plt.xticks(np.arange(1, result_df.shape[1] + 1), result_df.columns, rotation=45)
plt.ylabel("Acurácia")
plt.title("Comparação de Desempenho dos Modelos")
plt.show()

# Mostra média ± desvio
formatted_results = result_df.apply(lambda x: "{:.2f} ± {:.2f}".format(x.mean(), x.std()))
print("\nResultados (acurácia) com média ± desvio:")
print(formatted_results)

# Exemplo de teste de Wilcoxon entre dois modelos
stat, p_value = wilcoxon(result_df['tree'], result_df['bigtree'])
print("\nTeste de Wilcoxon entre kNN e tree:")
print("Estatística = {:.3f}, p-value = {:.3f}".format(stat, p_value))

# Teste de Friedman
friedman_stat, friedman_p = friedmanchisquare(*[result_df[col] for col in result_df.columns])
print("\nTeste de Friedman:")
print(f"  Estatística = {friedman_stat:.3f}, p-value = {friedman_p:.3f}")

# Teste post-hoc (Nemenyi) após Friedman
nemenyi_results = sp.posthoc_nemenyi_friedman(result_df)
print("\nMatriz de p-valores do Teste de Nemenyi:")
print(nemenyi_results)

# Ranks médios
avg_ranks = result_df.rank(axis=1, ascending=False, method='average').mean().sort_values()
print("\nRanks médios de cada algoritmo:")
print(avg_ranks)

# Cálculo do Critical Difference (CD)
k = result_df.shape[1]  # número de algoritmos
N = result_df.shape[0]  # número de folds
q_alpha = 2.728         # valor aproximado para alpha=0.05, k=6
cd = q_alpha * math.sqrt(k * (k + 1) / (6 * N))
print(f"\nCritical Difference (CD) = {cd:.3f}")

# Diagrama de Diferença Crítica simplificado
plt.figure(figsize=(10, 3))
algorithms_sorted = avg_ranks.index.tolist()
ranks_sorted = avg_ranks.values
y_rank = 1.0

# Pontos
plt.plot(ranks_sorted, [y_rank]*len(ranks_sorted), 'o', markersize=10, color='black')
for i, alg in enumerate(algorithms_sorted):
    plt.text(ranks_sorted[i], y_rank + 0.07, alg, rotation=45, ha='left', va='bottom')

plt.xlabel('Rank Médio')
plt.title('Diagrama de Critical Difference')
y_cd = 0.5
x_start = min(ranks_sorted)
plt.hlines(y_cd, x_start, x_start + cd, colors='red', linewidth=3)
plt.text(x_start + cd/2, y_cd - 0.1, f'CD = {cd:.2f}', color='red', ha='center')
plt.xlim(x_start - 0.5, max(ranks_sorted) + 0.5)
plt.ylim(0, y_rank + 0.5)
plt.yticks([])
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Curva ROC para cada modelo (usando todo X, y)
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
for alg_name, grid in algorithms.items():
    # Treina (fit) no conjunto todo (X, y)
    grid.fit(X, y)
    best_clf = grid.best_estimator_
    # Obtém scores para a classe positiva (1)
    if hasattr(best_clf, "predict_proba"):
        y_scores = best_clf.predict_proba(X)[:, 1]
    elif hasattr(best_clf, "decision_function"):
        y_scores = best_clf.decision_function(X)
    else:
        print(f"{alg_name} não suporta predição de probabilidades.")
        continue
    
    fpr, tpr, thresholds = roc_curve(y, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{alg_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="AUC = 0.5")
plt.xlabel("Falso Positivo (FPR)")
plt.ylabel("Verdadeiro Positivo (TPR)")
plt.title("Curvas ROC dos Modelos")
plt.legend(loc="lower right")
plt.show()

# -----------------------------------------------------------------------------
# Exemplo de "deploy" (treinando um modelo final e fazendo previsões)
# -----------------------------------------------------------------------------

# Escolhe um modelo
model_choice = 'tree'
classifier = algorithms[model_choice]

# Treina no dataset completo
classifier.fit(X, y)
print(f"\nMelhor estimador ({model_choice}) encontrado:")
print(classifier.best_estimator_)

# Faz previsões em cima de X_teste (por exemplo)
y_pred = classifier.predict(X_teste)
acc = accuracy_score(y_teste, y_pred)
print(f"\nAcurácia em X_teste: {acc:.4f}")

print("\nScript finalizado!")
