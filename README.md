# [Private and interpretable clinical prediction with quantum-inspired tensor train models](https://arxiv.org/abs/2602.06110)

**José Ramón Pareja Monturiol, Juliette Sinnott, Roger G. Melko, Mohammad Kohandel**

This repository contains the code used to run the experiments in the paper,
including training LR models, tensorizing such models, and training DP LR models.


## Versions used in the experiments

**Python**
- python == 3.11.11  

**Tensor network models**
- tensorkrowch == 1.1.5  

**Differentially private LR models**
- diffprivlib == 0.6.6

**Differentially private NN models**
- opacus == 1.5.4

**Machine learning packages**
- torch == 2.6.0  
- numpy == 2.1.2  
- scikit-learn == 1.7.1  

**Visualization packages**
- matplotlib  
- seaborn  

> **Note:** To render figure texts with $\LaTeX$, ensure that a LaTeX
            distribution is installed on your system.  


## Instructions

- **Privacy experiments**  
  See the `guide` file in the `privacy/` folder for instructions on training
  all model variants and collecting results. All scripts should be executed
  from the repository’s root directory. Results can be analyzed and visualized
  in the `attacks.ipynb` notebook.  

- **Interpretability experiments**  
  Conducted in the `interpretability.ipynb` notebook, located in the
  `interpretability/` folder.  


## Citing

If you would like to cite this work, please use the following format:

- J. R. Pareja Monturiol, J. Sinnott, R. G. Melko, M. Kohandel, *Private and 
interpretable clinical prediction with quantum-inspired tensor train models*, arXiv:2602.06110 (2026).

```
@misc{pareja2026private,
  title={Private and interpretable clinical prediction with quantum-inspired tensor train models}, 
  author={Pareja Monturiol, José Ramón and Sinnott, Juliette and Melko, Roger G. and Kohandel, Mohammad},
  year={2026},
  eprint={2602.06110},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2602.06110}, 
}
```
