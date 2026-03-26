# 🩺 Risco Doença 

Este projeto consiste em uma solução de ponta a ponta (End-to-End) para a análise e predição de risco cardiovascular utilizando técnicas de Machine Learning, monitoramento de experimentos com MLflow e deploy de uma interface interativa com Streamlit.

> **Aviso :** Este conteúdo é destinado apenas para fins educacionais. Os dados exibidos são ilustrativos e podem não corresponder a situações reais.

---

## 📂 Estrutura do Projeto

O repositório está organizado da seguinte forma:

| Arquivo | Descrição |
| :--- | :--- |
| `data.py` | Script responsável pelo carregamento e processamento inicial dos dados brutos. |
| `data_processed.csv` | Dataset resultante do processamento, pronto para consumo pelos modelos. |
| `eda.py` | Scripts de **Análise Exploratória de Dados**, contendo visualizações e estatísticas descritivas. |
| `main.py` | Script principal de treinamento. Implementa pipelines de ML e integração com o **MLflow**. |
| `mlflow.db` | Banco de dados local (SQLite) que armazena os metadados, parâmetros e métricas de cada experimento. |
| `modelo_final_saude.pkl` | O "artefato" do melhor modelo (Random Forest) serializado após validação no MLflow. |
| `app.py` | Interface web interativa desenvolvida em **Streamlit** para consumo do modelo. |
| `requirements.txt` | Lista de bibliotecas e dependências necessárias para a execução do projeto. |

---

## ⚙️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Machine Learning:** Scikit-Learn (Random Forest, Isolation Forest, PCA)
* **Métricas & Logs:** MLflow
* **Interface:** Streamlit
* **Manipulação de Dados:** Pandas, Numpy
* **Visualização:** Seaborn, Matplotlib

---

## 🚀 Como Executar o Projeto

### 1. Instalação de Dependências
Certifique-se de ter o Python instalado e rode:
```bash
pip install -r requirements.txt
