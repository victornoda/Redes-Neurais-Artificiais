# 🧠 House Price Prediction with Neural Networks

---

## 📖 Project Overview

This project details the creation of a neural network to predict housing prices using the well-known **"House Prices" dataset from Kaggle**. The primary goal is to leverage machine learning to accurately estimate the sale price of residential properties from a set of features.

The entire process is documented in a Jupyter Notebook (`TrabalhoRNA.ipynb`) and follows these key steps:

1.  **Data Loading & EDA**: We begin by loading the dataset and performing an Exploratory Data Analysis (EDA) to understand the features and the distribution of the target variable, `SalePrice`.
2.  **Simplified Preprocessing**: The data is cleaned by handling missing values (median for numbers, mode for categories) and encoding categorical features using `LabelEncoding`. To better model the target, `SalePrice` is transformed with `np.log1p`.
3.  **Data Splitting & Normalization**: The dataset is divided into training and validation sets, and all numerical features are normalized using `StandardScaler`.
4.  **Neural Network Architecture**: A `SimpleNeuralNetwork` class is defined in **PyTorch**, featuring linear layers, ReLU activation, and Dropout for regularization.
5.  **Model Training**: A `train_model` function orchestrates the training loop, utilizing the Adam optimizer and Mean Squared Error (`MSELoss`) as the loss function.
6.  **Hyperparameter Tuning**: We run a series of experiments with different hyperparameters (e.g., learning rate, epochs, network architecture, dropout) to identify the optimal model configuration.
7.  **Results Analysis**: The performance of each experiment is compared using key metrics like **R²** and **RMSE**. The training history (loss, R², RMSE) is visualized with plots.
8.  **Prediction & Submission**: The best-performing model is saved and used to generate predictions on the Kaggle test set, producing a `submission.csv` file.

---

## 🚀 Getting Started

### **1. Environment Setup**

* This project was developed and tested using **Python 3.11.4**.
* Using a virtual environment (like `venv` or `conda`) is highly recommended.

### **2. Required Libraries**

* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `tqdm`
* `torch` (PyTorch)
* `scikit-learn`

You can install all dependencies with a single command:

```bash
pip install pandas numpy matplotlib seaborn tqdm torch scikit-learn
```

### **3. Dataset**

Ensure the `train.csv` and `test.csv` files from the Kaggle competition are located in the same directory as your notebook.

### **4. Execution**

* Open and run the `TrabalhoRNA.ipynb` notebook in a Jupyter environment.
* Execute the cells sequentially.
* A random seed (`SEED = 42`) is used to ensure the results are reproducible.

### **5. Project Outputs**

A new directory named `resultados_modelos/` will be created to store:

* `comparacao_modelos.csv`: A summary table of performance metrics for all models.
* `comparacao_visual.png`: Comparative plots of the training curves.
* `melhor_modelo_[timestamp].pth`: The saved weights of the best model.
* `submissao_kaggle_[timestamp].csv`: The final prediction file for Kaggle submission.

---

## 🏆 Best Model Performance

The notebook systematically tests various configurations to find the most effective model. The best model, named `"modelo_l"`, was identified based on the highest **R² Score** on the validation data.

Its optimal hyperparameters were:

* **Architecture (`hidden_sizes`)**: `[128, 64]`
* **Learning Rate (`learning_rate`)**: `0.01`
* **Epochs (`num_epochs`)**: `500`
* **Dropout Rate (`dropout_rate`)**: `0.2`
* **Weight Decay (`weight_decay`)**: `0.0005`

---

## 💡 Additional Notes

* **PyTorch** was the exclusive autograd library used for this project.
* The model training was exceptionally efficient, with the best model training in just **1.6 seconds**.
* The code is structured with modular functions for preprocessing and training to enhance readability and maintainability.
* For simplicity, the entire training dataset is loaded into memory at once rather than being processed in batches.
