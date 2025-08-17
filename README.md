# Option Pricing with Deep Learning

## Overview
This project explores the use of **deep neural networks** to approximate the implied volatility surface and option prices, based on parameters from stochastic volatility models.  
The goal is to demonstrate how modern **Machine Learning** techniques can be applied in financial markets to replace or accelerate traditional numerical methods (such as Monte Carlo simulations or PDE solvers).  

We tested several neural network architectures (MLP, CNN, with normalization and regularization) to map volatility model dynamics into option prices.  

---

## 📂 Project Structure
- **`Main.ipynb`** : Main notebook for training and evaluating the pricing models.  
- **`Data_generator.ipynb`** : Synthetic data generation (simulation of stochastic models such as Bergomi).
- **`NN_trainings.ipynb`** : for testing different models. 
- **`Old models/`** : Contains different architectures (MLP, CNN, CNN optimized with Dropout and BatchNorm). 

---

## Models Implemented
### 1. **MLP – Implied Volatility Surface Approximation**
A multi-layer perceptron with several hidden layers (30 neurons, *tanh* activation) to directly approximate the implied volatility surface.  

### 2. **CNN – Option Pricing as Image Recognition**
The implied volatility surface is transformed into an "image" (strike × maturity grid), enabling **Convolutional Neural Networks (CNNs)** to capture its spatial structure.  

- `model_cnn` : basic version with 2D convolutions.  
- `model_cnn_tuned` : improved version with **Batch Normalization** and **Dropout** for stability and regularization.  

### 3. **MLP + Batch Normalization**
A second MLP, enhanced with **Batch Normalization** between layers and explicit activations, for improved stability and faster convergence.  

---

## Methodology
1. **Data Generation** : Simulating option prices under stochastic volatility models (here, the Bergomi model).  
2. **Preprocessing** : Converting simulated data into representations usable by our models (implied volatility vectors or strike–maturity grids).  
3. **Model Training** :  
   - Framework: Keras/TensorFlow.  
   - Optimizer: *Adam*.  
   - Loss function: *Mean Squared Error (MSE)*.  
4. **Evaluation** : Comparing model architectures on pricing accuracy and surface reconstruction quality.  

---

## Why is this interesting for Finance?
- **Speed** : Once trained, neural networks provide instant predictions of implied volatility surfaces, bypassing costly numerical methods.  
- **Accuracy** : CNNs capture local structures in volatility surfaces, improving robustness.
- **Innovation** : Combining financial simulations with deep learning is a cutting-edge research direction in *quantitative finance*.  

---

## 🔧 Tech Stack
- Python (3.13)  
- TensorFlow / Keras  
- Numpy, Pandas, Matplotlib, Seaborn  

