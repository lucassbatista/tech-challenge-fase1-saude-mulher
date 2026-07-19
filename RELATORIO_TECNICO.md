# Relatório Técnico — Tech Challenge Fase 1

## Saúde e Segurança da Mulher — Diagnóstico de Câncer de Mama

## 1. Contexto e problema

Uma rede de hospitais e centros de saúde especializados no atendimento à mulher busca
implementar um sistema inteligente de apoio ao diagnóstico e à detecção precoce de
condições que afetam a saúde e segurança feminina. Nesta primeira fase, o objetivo foi
construir a base do sistema de IA/Machine Learning: um pipeline completo de
processamento de dados médicos, treinamento de modelos preditivos e avaliação crítica
dos resultados, usando como caso de uso o **diagnóstico de câncer de mama** (maligno x
benigno).

Foi desenvolvido o **modelo principal** com dados estruturados —
[`analise_cancer_mama.ipynb`](notebooks/analise_cancer_mama.ipynb) — cobrindo todo o
pipeline de ciência de dados pedido no desafio.

---

## 2. Modelo principal — dados estruturados

### 2.1 Dataset

**Breast Cancer Wisconsin (Diagnostic)** — 569 exames, 30 variáveis numéricas extraídas
de imagens digitalizadas de punção aspirativa por agulha fina (FNA) de massas mamárias
(ex.: raio, textura, perímetro, área, suavidade, compacidade, concavidade, simetria —
cada uma com média, erro padrão e "pior" valor). Carregado via
`sklearn.datasets.load_breast_cancer` para garantir reprodutibilidade sem dependência de
download manual (mesma base disponível no [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)).

### 2.2 Análise exploratória

- **Sem valores ausentes** nas 32 colunas (569 linhas × 30 features + rótulo).
- **Distribuição das classes:** 62,7% benignos e 37,3% malignos — desbalanceamento
  moderado, relevante para a escolha das métricas de avaliação (seção 2.5).
- **Padrões visuais:** comparando `mean radius`, `mean perimeter`, `mean area` e
  `mean smoothness` entre as classes, tumores malignos apresentam consistentemente
  valores maiores e maior dispersão — coerente com o conhecimento clínico de que
  tumores malignos tendem a ser maiores e mais irregulares que os benignos.

### 2.3 Pré-processamento

1. Verificação de dados ausentes/inconsistentes — nenhum encontrado, dispensando
   imputação;
2. Todas as variáveis já são numéricas — não há variáveis categóricas para codificar;
3. Separação treino/teste (80/20) **estratificada** pela classe, para preservar a
   proporção de malignos/benignos em ambos os conjuntos (455 amostras treino, 114
   teste);
4. **Padronização** (`StandardScaler`) de todas as features — necessária para modelos
   sensíveis à escala das variáveis, como Regressão Logística e KNN (a Árvore de Decisão
   não exige, mas manter o mesmo pré-processamento simplifica o pipeline).

### 2.4 Análise de correlação

A matriz de correlação revelou forte colinearidade entre variáveis de tamanho (`radius`,
`perimeter`, `area` — como esperado, pois são medidas derivadas geometricamente umas das
outras). As variáveis mais correlacionadas com o diagnóstico foram:

| Variável | Correlação com o alvo |
|---|---|
| `worst concave points` | -0.79 |
| `worst perimeter` | -0.78 |
| `mean concave points` | -0.78 |
| `worst radius` | -0.78 |
| `mean perimeter` | -0.74 |

(Correlação negativa porque `target=0` representa maligno na codificação do
scikit-learn.) Isso indica forte poder discriminativo de variáveis relacionadas à
irregularidade de contorno e ao tamanho do tumor — confirmado depois pela análise de
explicabilidade (seção 2.6).

### 2.5 Modelagem, treinamento e escolha de métricas

Foram treinadas três técnicas de classificação, escolhidas por representarem
abordagens distintas (linear, baseada em árvore, e baseada em distância):

| Modelo | Motivação |
|---|---|
| Regressão Logística | Baseline linear, simples, interpretável e compatível com SHAP |
| Árvore de Decisão (`max_depth=5`) | Captura relações não-lineares e interações entre variáveis |
| KNN (`k=5`) | Classificação por similaridade entre pacientes, abordagem não-paramétrica |

**Escolha da métrica:** em diagnóstico médico, um **falso negativo** (classificar um
tumor maligno como benigno) tem consequência muito mais grave que um falso positivo,
pois pode atrasar o tratamento. Por isso, priorizamos o **recall da classe maligna**
como métrica principal, complementado pelo **F1-score** (equilíbrio entre precisão e
recall). A acurácia isoladamente pode ser enganosa dado o desbalanceamento das classes.

### 2.6 Resultados

| Modelo | Accuracy | Recall (maligno) | F1-score (maligno) | ROC-AUC |
|---|---|---|---|---|
| **Regressão Logística** | 0.98 | **0.98** | 0.98 | 0.995 |
| Árvore de Decisão | 0.92 | 0.93 | 0.90 | 0.916 |
| KNN | 0.96 | 0.93 | 0.94 | 0.979 |

*(Recall/F1 da tabela referem-se à classe "maligno", a mais crítica clinicamente.)*

A **Regressão Logística** apresentou o melhor desempenho geral, com recall de 0.98 para
a classe maligna — ou seja, deixou de identificar corretamente apenas um caso maligno em
42 no conjunto de teste. A Árvore de Decisão teve o pior desempenho relativo, esperado
dado sua tendência a overfitting em datasets pequenos, mesmo com profundidade limitada.

### 2.7 Explicabilidade

- **Feature importance (Árvore de Decisão):** as variáveis mais relevantes para as
  divisões da árvore foram relacionadas a tamanho e concavidade do tumor
  (`worst perimeter`, `worst concave points`, `worst radius`, `mean concave points`).
- **SHAP (Regressão Logística):** o summary plot confirma o mesmo padrão — variáveis de
  concavidade e tamanho no "pior caso" (`worst`) têm o maior impacto absoluto na
  predição, com valores altos empurrando a previsão para "maligno".

Essa convergência entre os dois métodos de explicabilidade reforça a robustez do sinal
— e é clinicamente coerente, já que irregularidade de contorno e tamanho são indicadores
conhecidos de malignidade em exames de imagem.

### 2.8 Discussão crítica

O modelo apresentou métricas fortes, mas algumas limitações devem ser destacadas antes
de qualquer uso prático:

- **Tamanho e origem do dataset:** apenas 569 amostras de uma única instituição, o que
  limita a generalização para outras populações, equipamentos e protocolos de exame;
- **Desbalanceamento:** ainda que moderado, reforça a necessidade de monitorar recall
  da classe minoritária (maligno) continuamente, inclusive após deploy;
- **Uso responsável:** o modelo deve ser utilizado como **ferramenta de apoio à
  triagem**, ajudando a priorizar casos suspeitos para revisão médica mais rápida — **o
  médico deve sempre ter a palavra final no diagnóstico**. O sistema não substitui
  biópsia, exames complementares ou julgamento clínico.

---

## 3. Conclusão e próximos passos

A Fase 1 entrega a base funcional do sistema de IA: pipeline reprodutível de
exploração, pré-processamento, modelagem, avaliação e explicabilidade para dados
estruturados de diagnóstico de câncer de mama. Os próximos passos naturais são:

- Incorporar dados de imagem (mamografias) com redes neurais convolucionais (CNN),
  utilizando o dataset CBIS-DDSM sugerido no desafio;
- Ampliar a base estruturada com dados de outras instituições, reduzindo o risco de
  overfitting a um único protocolo de exame;
- Investigar outros temas de saúde e segurança da mulher sugeridos no desafio (ex.:
  Síndrome dos Ovários Policísticos, sinais de violência doméstica em prontuários);
- Sempre manter o médico como responsável final pelo diagnóstico, com o modelo atuando
  apenas como ferramenta de apoio à triagem.
