# Dataset

Este projeto utiliza o dataset **Breast Cancer Wisconsin (Diagnostic)**.

- Fonte original (Kaggle): https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- No notebook, o dataset é carregado diretamente via `sklearn.datasets.load_breast_cancer`,
  que disponibilza a mesma base (UCI Machine Learning Repository), garantindo
  reprodutibilidade sem necessidade de download manual ou credenciais do Kaggle.

Caso deseje utilizar o CSV baixado do Kaggle em vez da versão embutida no scikit-learn,
salve o arquivo `data.csv` nesta pasta e adapte a célula de carregamento de dados no
notebook (`notebooks/analise_cancer_mama.ipynb`).
