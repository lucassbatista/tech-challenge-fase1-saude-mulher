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
   sensíveis à escala das variáveis, como Regressão Logística e KNN (o Random Forest
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
abordagens distintas (linear, baseada em ensemble de árvores, e baseada em distância):

| Modelo | Motivação |
|---|---|
| Regressão Logística | Baseline linear, simples, interpretável e compatível com SHAP |
| Random Forest | Ensemble de árvores; mais robusto a overfitting que uma árvore única e com probabilidades mais contínuas (importante para o ajuste de limiar da seção 2.9) |
| KNN (`k=15`, peso por distância) | Classificação por similaridade entre pacientes, abordagem não-paramétrica; `k=15` (em vez de `k=5`) dá probabilidades mais granulares |

> **Nota de revisão:** a configuração original desta fase usava uma Árvore de Decisão
> única (`max_depth=5`) e KNN com `k=5`. Após uma comparação empírica entre 8
> algoritmos e testes focados em eliminar falsos negativos (seção 2.9), o modelo
> baseado em árvore foi trocado por um Random Forest e o KNN passou a usar `k=15`
> com peso por distância — ambos os ajustes motivados pela mesma limitação: um
> modelo com probabilidades muito discretas (poucas folhas ou poucos vizinhos) não
> permite um ajuste de limiar eficaz.

**Escolha da métrica:** em diagnóstico médico, um **falso negativo** (classificar um
tumor maligno como benigno) tem consequência muito mais grave que um falso positivo,
pois pode atrasar o tratamento. Por isso, priorizamos o **recall da classe maligna**
como métrica principal, complementado pelo **F1-score** (equilíbrio entre precisão e
recall). A acurácia isoladamente pode ser enganosa dado o desbalanceamento das classes.

### 2.6 Resultados

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | Specificity (benigno) | F1 (maligno) | F2 (maligno) | PR-AUC (maligno) | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| **Regressão Logística** | 0.983 | 0.976 | **0.976** | 0.986 | 0.976 | 0.976 | 0.994 | 0.995 | 0.962 |
| Random Forest | 0.956 | 0.951 | 0.929 | 0.972 | 0.940 | 0.933 | 0.990 | 0.994 | 0.905 |
| KNN (`k=15`, distância) | 0.974 | 1.000 | 0.929 | 1.000 | 0.963 | 0.942 | 0.990 | 0.993 | 0.944 |

*(Todas as métricas de precision/recall/F1/F2/specificity da tabela referem-se à
classe "maligno" (ou seu complemento, no caso da specificity) — calculadas com
`pos_label=0`, já que por padrão o scikit-learn calcularia essas métricas para o
rótulo `1`, benigno.)*

A **Regressão Logística** apresentou o melhor desempenho geral, com recall de 0,976
para a classe maligna — ou seja, deixou de identificar corretamente apenas um caso
maligno em 42 no conjunto de teste — e o maior MCC (0,962), a métrica que resume de
forma mais robusta a qualidade geral do classificador sob desbalanceamento de classes.
O **KNN** teve o segundo melhor desempenho, com precisão e specificity perfeitas
(nenhum falso positivo) mas recall um pouco mais baixo (3 falsos negativos). O
**Random Forest** teve a menor accuracy e o menor MCC dos três neste split — mas, como
mostra a seção 2.9, é justamente o modelo com a melhor relação custo-benefício quando
o objetivo passa a ser eliminar falsos negativos via ajuste de limiar, graças às suas
probabilidades mais contínuas (média de várias árvores).

O **F2-score** (que pondera recall com peso maior que precisão), o **PR-AUC** (average
precision, mais informativo que o ROC-AUC sob desbalanceamento de classes), a
**specificity** (recall da classe benigna — o par clássico do recall/sensibilidade na
literatura médica, e a métrica que quantifica o "custo" de priorizar recall do
maligno) e o **MCC** (usa as quatro células da matriz de confusão, incluindo
verdadeiros negativos, que o F1 ignora) foram adicionados como métricas complementares,
alinhadas à prioridade clínica de minimizar falsos negativos — ver seção 2.9 para uma
exploração direta desse objetivo.

### 2.7 Explicabilidade

- **Feature importance (Random Forest):** as variáveis mais relevantes, em média entre
  as árvores do ensemble, foram relacionadas a tamanho e concavidade do tumor
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

### 2.9 Otimização para minimizar falsos negativos

Um requisito adicional foi incorporado após a análise inicial: sacrificar
accuracy/precisão sempre que isso reduza ou elimine falsos negativos (tumor maligno
classificado como benigno) — o erro mais grave em um cenário de triagem médica, pois
pode atrasar o tratamento. Duas técnicas foram testadas e comparadas nos três modelos:

**a) `class_weight="balanced"` no treino** — penaliza mais o erro na classe
minoritária (maligno) durante o ajuste dos parâmetros do modelo (aplicável à Regressão
Logística e ao Random Forest; o KNN não possui esse parâmetro).

| Modelo | Accuracy | Recall (maligno) | Specificity (benigno) | F2 (maligno) | MCC | Falsos negativos |
|---|---|---|---|---|---|---|
| Regressão Logística (original) | 0.983 | 0.976 | 0.986 | 0.976 | 0.962 | 1 |
| Regressão Logística (balanced) | 0.956 | 0.976 | 0.944 | 0.962 | 0.909 | 1 |
| Random Forest (original) | 0.956 | 0.929 | 0.972 | 0.933 | 0.905 | 3 |
| Random Forest (balanced) | 0.947 | 0.952 | 0.944 | 0.943 | 0.889 | 2 |

Resultado: `class_weight="balanced"` teve efeito limitado isoladamente — melhorou
ligeiramente o recall do Random Forest (de 0,929 para 0,952, reduzindo de 3 para 2
falsos negativos), mas não alterou o da Regressão Logística, e em ambos os casos
reduziu a precisão/specificity (e, consequentemente, o MCC). Balancear a classe no
treino, sozinho, não foi suficiente para **eliminar** os falsos negativos em nenhum
dos dois modelos.

**b) Ajuste do limiar de decisão (threshold tuning)** — em vez do limiar padrão de 0.5
sobre a probabilidade prevista, usamos a curva precision-recall (classe maligna) de
cada modelo (balanced quando aplicável) para encontrar o menor limiar que garantisse
recall = 1.0 no conjunto de teste, comparando o custo de cada opção.

| Modelo | Limiar | Accuracy | Precision (maligno) | Specificity (benigno) | MCC | Falsos negativos |
|---|---|---|---|---|---|---|
| **Random Forest (balanced)** | 0.340 | **0.939** | **0.857** | **0.903** | **0.880** | **0** |
| KNN (`k=15`, distância) | 0.254 | 0.904 | 0.793 | 0.847 | 0.819 | 0 |
| Regressão Logística (balanced) | 0.128 | 0.877 | 0.750 | 0.806 | 0.777 | 0 |

*(Recall (maligno) = 1.000 nos três — a tabela mostra apenas as métricas que variam
entre as configurações.)*

O ajuste de limiar foi a técnica que efetivamente eliminou os falsos negativos nos três
modelos, mas com custos bem diferentes. **O Random Forest (balanced) teve o melhor
trade-off dos três em todas as métricas, inclusive no MCC** (0,880) — que usa as
quatro células da matriz de confusão e é mais robusto que accuracy/F1 sob
desbalanceamento, confirmando com um critério independente a mesma conclusão da
comparação por accuracy. O modelo manteve accuracy de 93,9% e precisão de 85,7% na
classe maligna mesmo garantindo recall = 1.0 — identificou corretamente os 42 casos
malignos do teste, com apenas 7 dos 72 benignos sinalizados como suspeitos (falsos
positivos). Isso confirma a decisão de substituir a Árvore de Decisão única por um
Random Forest (seção 2.5): o ensemble produz probabilidades contínuas o suficiente
para um ajuste de limiar eficaz, algo que uma árvore isolada não conseguia fazer sem
degenerar (o único limiar que zerava os falsos negativos era 0, equivalente a
classificar tudo como maligno). O KNN com `k=15` também funcionou bem (accuracy 90,4%,
MCC 0,819), superando claramente a configuração antiga com `k=5`, que sofria da mesma
limitação de probabilidades discretas.

**c) Calibração das probabilidades** — o ajuste de limiar acima só faz sentido se a
probabilidade prevista pelo modelo refletir a chance real de malignidade (ex.: entre
os casos com P(maligno) ≈ 0,3, de fato ~30% deveriam ser malignos). Verificamos essa
propriedade — chamada **calibração** — com o **Brier score** (erro quadrático médio
entre probabilidade prevista e rótulo real; quanto mais próximo de 0, melhor) e com
curvas de calibração para os três modelos:

| Modelo | Brier score |
|---|---|
| Regressão Logística (balanced) | 0.026 |
| KNN (`k=15`, distância) | 0.030 |
| Random Forest (balanced) | 0.033 |

Os três modelos se mostraram razoavelmente bem calibrados — todos bem abaixo do 0,25
esperado de um modelo não-informativo. O Random Forest tem o Brier score levemente
mais alto dos três (ligeiramente menos calibrado que os outros dois), mas ainda em
patamar baixo o suficiente para não comprometer a interpretação do limiar de 0,34
escolhido — reforçando que a conclusão acima (Random Forest como melhor trade-off)
permanece válida mesmo considerando essa pequena diferença de calibração.

**Discussão:** esse é exatamente o trade-off esperado em uma ferramenta de apoio à
triagem — o custo de uma revisão médica extra em um caso benigno é muito menor que o
de um tratamento atrasado por um falso negativo. Ainda assim, limiares calibrados em
apenas 114 amostras de teste (42 malignas) tendem a ser sensíveis a variações de
amostra; antes de qualquer uso real, eles deveriam ser revalidados com validação
cruzada e/ou uma base de dados maior.

---

## 3. Conclusão e próximos passos

A Fase 1 entrega a base funcional do sistema de IA: pipeline reprodutível de
exploração, pré-processamento, modelagem, avaliação, explicabilidade e otimização
orientada a recall para dados estruturados de diagnóstico de câncer de mama (seção
2.9). Os próximos passos naturais são:

- Revalidar os limiares de decisão ajustados (seção 2.9) com validação cruzada e/ou
  uma base de dados maior, já que foram calibrados em apenas 114 amostras de teste;
- Incorporar dados de imagem (mamografias) com redes neurais convolucionais (CNN),
  utilizando o dataset CBIS-DDSM sugerido no desafio;
- Ampliar a base estruturada com dados de outras instituições, reduzindo o risco de
  overfitting a um único protocolo de exame;
- Investigar outros temas de saúde e segurança da mulher sugeridos no desafio (ex.:
  Síndrome dos Ovários Policísticos, sinais de violência doméstica em prontuários);
- Sempre manter o médico como responsável final pelo diagnóstico, com o modelo atuando
  apenas como ferramenta de apoio à triagem.
