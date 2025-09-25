# Private and Interpretable Clinical Prediction with Quantum-Inspired Tensor Train Models

This repository contains the code used to run the experiments in the paper,
including training LR models, tensorizing such models, and training DP LR models.

---

## Quick Start

Clone the repository:
```bash
git clone https://github.com/joserapa98/tns4loris.git
cd tns4loris
```

---

## Versions used in the experiments

**Python**
- python == 3.11.11  

**Tensor network models**
- tensorkrowch == 1.1.5  

**Differentially private LR models**
- diffprivlib == 0.6.6  

**Machine learning packages**
- torch == 2.6.0  
- numpy == 2.1.2  
- scikit-learn == 1.7.1  

**Visualization packages**
- matplotlib  
- seaborn  

> **Note:** To render figure texts with $\LaTeX$, ensure that a LaTeX
            distribution is installed on your system.  

---

## Instructions

- **Privacy experiments**  
  See the `guide` file in the `privacy/` folder for instructions on training
  all model variants and collecting results. All scripts should be executed
  from the repository’s root directory. Results can be analyzed and visualized
  in the `attacks.ipynb` notebook.  

- **Interpretability experiments**  
  Conducted in the `interpretability.ipynb` notebook, located in the
  `interpretability/` folder.  
