# Tech Challenge - Fase 1 - Saúde da Mulher

Sistema de IA focado em Machine Learning, permitindo que dados médicos sejam
analisados automaticamente para identificar padrões de risco relacionados à
segurança e saúde das mulheres.

## Desafio

Uma rede de hospitais e centros de saúde especializados no atendimento à mulher
busca implementar um sistema inteligente de suporte ao diagnóstico e detecção de
riscos, capaz de ajudar profissionais de saúde na identificação precoce de
condições que afetam a segurança e saúde feminina.

Nesta primeira fase, o desafio é criar a base do sistema de IA focado em Machine
Learning, permitindo que dados médicos sejam analisados automaticamente para
identificar padrões de risco relacionados à segurança e saúde das mulheres.

## Solução desta fase

Foi desenvolvido um modelo de classificação para apoiar o **diagnóstico de câncer
de mama** (maligno x benigno), utilizando o dataset **Breast Cancer Wisconsin
(Diagnostic)**. O notebook cobre todo o pipeline de ciência de dados:

1. Exploração de dados (estatísticas descritivas, distribuições, padrões);
2. Pré-processamento (checagem de dados ausentes, padronização, split treino/teste);
3. Análise de correlação entre variáveis;
4. Modelagem com múltiplos algoritmos (Regressão Logística, Árvore de Decisão, KNN);
5. Treinamento e avaliação (accuracy, precision, recall, F1-score, ROC-AUC);
6. Explicabilidade (feature importance e SHAP);
7. Discussão crítica sobre uso prático do modelo.

> ⚠️ O modelo é uma ferramenta de **apoio à triagem** e não substitui a avaliação
> clínica. O médico deve sempre ter a palavra final no diagnóstico.

## Estrutura do projeto

```
.
├── data/                            # informações sobre os datasets utilizados
├── notebooks/
│   └── analise_cancer_mama.ipynb    # notebook principal (dados estruturados)
├── requirements.txt
├── Dockerfile
├── RELATORIO_TECNICO.md
└── README.md
```

## Como executar

### 1. Criar e ativar ambiente virtual

**Windows:**

```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar o notebook

```bash
jupyter notebook notebooks/analise_cancer_mama.ipynb
```

### 4. Desativar ambiente virtual

```bash
deactivate
```

## Dataset

- **Breast Cancer Wisconsin (Diagnostic)** — [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- Carregado diretamente via `sklearn.datasets.load_breast_cancer` para garantir
  reprodutibilidade. Mais detalhes em [data/README.md](data/README.md).

## Executando com Docker

```bash
docker build -t tech-challenge-saude-mulher .
docker run -p 8888:8888 -v "${PWD}:/app" tech-challenge-saude-mulher
```

O comando acima sobe um servidor Jupyter em `http://localhost:8888`.

## Relatório técnico

O relatório técnico completo (análise exploratória, pré-processamento, modelos e
interpretação dos resultados) está em [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md).

## Próximos passos

- Incorporar dados de imagem (mamografias) com redes neurais convolucionais (CNN);
- Avaliar outras bases relacionadas à saúde e segurança da mulher (ex.: SOP);
- Expandir a documentação técnica com relatório completo para a entrega da Fase 1.
