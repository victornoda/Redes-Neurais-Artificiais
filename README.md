# Predição de Preços de Casas com Redes Neurais

## Trabalho RNA

**Alunos:** Victor Shin Iti Kanazawa Noda, Lara Portilho Marques, João Victor Araujo Costa

## Descrição do Projeto

Este projeto apresenta a implementação de redes neurais para a predição de preços de casas, utilizando o dataset "House Prices" do Kaggle. O objetivo é aplicar técnicas de aprendizado de máquina para estimar o valor de venda de imóveis residenciais a partir de suas características. O notebook Jupyter (`TrabalhoRNA.ipynb`) inclui as seguintes etapas:

1.  **Carregamento e Análise Exploratória dos Dados**: Importação do dataset, visualização inicial e análise da distribuição da variável alvo (`SalePrice`).
2.  **Pré-processamento Simplificado**: Limpeza dos dados, preenchimento de valores ausentes (mediana para numéricos, moda para categóricos) e codificação de variáveis categóricas (Label Encoding). A variável alvo `SalePrice` é transformada com `np.log1p` para melhor distribuição.
3.  **Divisão dos Dados e Normalização**: Os dados são divididos em conjuntos de treino e validação. As features numéricas são normalizadas utilizando `StandardScaler`.
4.  **Definição da Rede Neural**: É definida uma classe `SimpleNeuralNetwork` em PyTorch, consistindo em camadas lineares com função de ativação ReLU e Dropout para regularização.
5.  **Função de Treinamento**: Uma função `train_model` é implementada para realizar o loop de treinamento, utilizando o otimizador Adam e a função de perda MSELoss.
6.  **Configuração e Execução de Experimentos**: São definidas e executadas múltiplas configurações de hiperparâmetros (taxa de aprendizado, número de épocas, arquitetura da rede, dropout, weight decay) para encontrar o melhor modelo.
7.  **Análise Comparativa e Visualização dos Resultados**: Os resultados dos experimentos são comparados com base em métricas como R² e RMSE. Gráficos das curvas de perda, R² e RMSE são gerados.
8.  **Salvamento do Melhor Modelo e Predições**: O melhor modelo é salvo e utilizado para gerar predições no dataset de teste do Kaggle, criando um arquivo CSV para submissão.

## Instruções de Execução

1.  **Ambiente**:
    * O código foi desenvolvido e testado em Python 3.11.4.
    * Recomenda-se o uso de um ambiente virtual (ex: venv, conda) para gerenciar as dependências.

2.  **Bibliotecas Usadas**:
    * pandas
    * numpy
    * matplotlib
    * seaborn
    * tqdm
    * torch (PyTorch)
    * scikit-learn (sklearn)
    * datetime (para timestamps nos arquivos de saída)
    * os (para manipulação de diretórios)

    Para instalar as dependências, você pode usar o pip:
    ```bash
    pip install pandas numpy matplotlib seaborn tqdm torch scikit-learn
    ```

3.  **Arquivos do Dataset**:
    * Certifique-se de que os arquivos `train.csv` e `test.csv` (do dataset "House Prices" do Kaggle) estejam no mesmo diretório do notebook Jupyter.

4.  **Execução do Notebook**:
    * Abra o arquivo `TrabalhoRNA.ipynb` em um ambiente Jupyter Notebook ou JupyterLab.
    * Execute as células na ordem em que aparecem.
    * Os hiperparâmetros para os diferentes experimentos estão definidos em uma célula específica (célula de código 15).
    * A semente de números aleatórios (`SEED = 42`) é configurada no início para garantir a reprodutibilidade.

5.  **Saídas**:
    * Uma pasta chamada `resultados_modelos/` será criada no mesmo diretório do notebook.
    * Dentro desta pasta, serão salvos:
        * `comparacao_modelos.csv`: Tabela com as métricas de todos os experimentos.
        * `comparacao_visual.png`: Gráficos comparando as curvas de perda, R² e RMSE dos experimentos.
        * `melhor_modelo_[timestamp].pth`: O estado do melhor modelo treinado.
        * `predicoes_melhor_modelo_[timestamp].csv`: Predições do melhor modelo no dataset de treino original.
        * `submissao_kaggle_[timestamp].csv`: Arquivo de submissão para o Kaggle com as predições no dataset de teste.

## Detalhes dos Experimentos e Melhor Modelo

O notebook testa várias configurações de rede neural, variando:
* Número de camadas ocultas e neurônios por camada.
* Taxa de aprendizado (`learning_rate`).
* Número de épocas (`num_epochs`).
* Taxa de dropout (`dropout_rate`).
* Decaimento de peso (`weight_decay`).

O melhor modelo é selecionado com base no R² Score no conjunto de validação. Os hiperparâmetros do melhor modelo encontrado ("modelo\_l") foram:
* **Arquitetura (`hidden_sizes`)**: [128, 64]
* **Taxa de Aprendizado (`learning_rate`)**: 0.01
* **Número de Épocas (`num_epochs`)**: 500
* **Dropout (`dropout_rate`)**: 0.2
* **Weight Decay (`weight_decay`)**: 0.0005

## Observações

* A única biblioteca de autograd permitida e utilizada foi o PyTorch.
* O treinamento do melhor modelo foi significativamente rápido (1.6 segundos), bem abaixo do limite de 1 hora.
* O código está modularizado em funções para pré-processamento e treinamento.
* O carregamento dos dados para o treinamento não é feito em batches, mas sim com o dataset de treino completo de uma vez.