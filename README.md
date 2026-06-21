# Molecular-Energy-Neural-Network

# Coulomb Matrix Based Molecular Energy Prediction

This project builds a machine learning pipeline for predicting molecular energy from molecular structure. The dataset contains JSON files of molecules scraped from PubChem, where each molecule includes an energy value, atom types, and three-dimensional atomic coordinates. The project performs feature engineering using the Coulomb matrix, applies memory-efficient dimensionality reduction with Incremental PCA, and trains regression models to predict molecular energy.

The final selected model is an `MLPRegressor` trained on a 1,400-component PCA representation of Coulomb matrix features. The best model achieved an R² score of approximately **0.8465** on the held-out test set.

---

## 1. Project Overview

Molecular simulations can be computationally expensive, especially when they are applied to large chemical databases containing millions of molecules. Machine learning can be used as a surrogate modeling approach, where a model learns the relationship between molecular structure and molecular properties from existing data.

In this project, molecular structures are converted into numerical features using the Coulomb matrix representation. The resulting feature matrix is high-dimensional and memory-intensive, so Incremental PCA is used to reduce dimensionality before model training.

The main workflow is:

```text
JSON molecular data
        ↓
Atom extraction and padding
        ↓
Atomic number mapping
        ↓
Coulomb matrix generation
        ↓
Feature flattening
        ↓
Standard scaling
        ↓
Incremental PCA
        ↓
Machine learning regression
        ↓
Energy prediction
```

---

## 2. Dataset

The dataset used in this project is available on Kaggle:

[Predict Molecular Properties Dataset](https://www.kaggle.com/datasets/burakhmmtgl/predict-molecular-properties/data)

The dataset contains molecular properties scraped from the PubChem database. Each molecule contains atom identities and atomic coordinates, making it suitable for feature engineering tasks in molecular machine learning.

### 2.1 Dataset Files Used

The following JSON files were used in this project:

```text
pubChem_p_00000001_00025000.json
pubChem_p_00025001_00050000.json
pubChem_p_00050001_00075000.json
pubChem_p_00075001_00100000.json
pubChem_p_00100001_00125000.json
pubChem_p_00125001_00150000.json
pubChem_p_00150001_00175000.json
pubChem_p_00175001_00200000.json
pubChem_p_00200001_00225000.json
pubChem_p_00225001_00250000.json
```

The combined dataset used in this project contained approximately **171,137 molecules**.

### 2.2 JSON Structure

Each JSON file contains a list of molecule objects. A simplified example is shown below:

```json
{
  "En": 37.801,
  "atoms": [
    {"type": "O", "xyz": [0.3387, 0.9262, 0.46]},
    {"type": "C", "xyz": [-0.7783, -1.1579, 0.0914]},
    {"type": "H", "xyz": [-0.7073, -2.1051, -0.4563]}
  ],
  "id": 1,
  "shapeM": [259.66, 4.28, 3.04, 1.21]
}
```

### 2.3 Fields

| Field    | Description                                            | Use in this project                     |
| -------- | ------------------------------------------------------ | --------------------------------------- |
| `En`     | Molecular energy calculated using a force-field method | Target variable                         |
| `atoms`  | List of atoms with element type and xyz coordinates    | Used for Coulomb matrix generation      |
| `type`   | Element symbol, such as H, C, N, O                     | Mapped to atomic number                 |
| `xyz`    | Cartesian coordinates of each atom                     | Used to calculate interatomic distances |
| `id`     | PubChem molecule identifier                            | Not used in model training              |
| `shapeM` | Shape multipole features                               | Not used in the final pipeline          |

The dataset contains molecules with different numbers of atoms. This makes feature engineering challenging because machine learning models generally require fixed-size numerical inputs.

---

## 3. Methodology

### 3.1 Molecular Representation

The main feature representation used in this project is the Coulomb matrix. For a molecule with atoms indexed by `i` and `j`, the Coulomb matrix is defined as:

```text
C_ij = Zi * Zj / |Ri - Rj|        if i != j

C_ii = 0.5 * Zi^2.4              if i == j
```

Where:

* `Zi` and `Zj` are atomic numbers.
* `Ri` and `Rj` are the Cartesian coordinate vectors of atoms `i` and `j`.
* `|Ri - Rj|` is the Euclidean distance between two atoms.

The off-diagonal elements represent pairwise nuclear interactions, while the diagonal elements encode atom-specific terms.

### 3.2 Fixed-Size Feature Construction

The largest molecule observed during preprocessing contained **121 atoms**. Therefore, each molecule was represented using a fixed Coulomb matrix of shape:

```text
121 × 121
```

After flattening, each molecule initially had:

```text
121 × 121 = 14,641 features
```

Smaller molecules were padded so that every molecule had the same feature dimension.

### 3.3 Memory Challenge

The full dataset produces a very large dense feature matrix:

```text
171,137 molecules × 14,641 features ≈ 2.51 billion values
```

Using `float64`, this would require approximately 18.7 GiB for the feature matrix alone. To reduce memory usage, the project used `float32` arrays for the main numerical operations.

### 3.4 Incremental PCA

Standard PCA was not practical because of the dataset size and memory constraints. Instead, `IncrementalPCA` was used to process the data in batches.

The final PCA configuration used:

```text
n_components = 1400
batch_size = 1400
```

The 1,400-component PCA representation retained approximately **94.9%** of the variance in the standardized Coulomb matrix features.

---

## 4. Models Tested

Several regression models were explored during development:

| Model                         | Outcome                                            |
| ----------------------------- | -------------------------------------------------- |
| Ridge Regression              | Fast baseline, but underfit the data               |
| Lasso Regression              | Fast baseline, but not strong enough for final use |
| Decision Tree Regressor       | Limited generalization                             |
| K-Nearest Neighbors           | Computationally expensive at full scale            |
| Random Forest Regressor       | Promising but slow on the available machine        |
| Gradient Boosting Regressor   | Useful baseline but computationally heavier        |
| HistGradientBoostingRegressor | Strong and efficient tabular baseline              |
| MLPRegressor                  | Best final performance                             |

The final model was selected based on a combination of predictive performance and computational feasibility.

---

## 5. Final Model

The best model was an `MLPRegressor` trained on the 1,400-component PCA feature matrix.

Final architecture:

```python
MLPRegressor(
    hidden_layer_sizes=(1024,512,256,128,64,64,28,64,16),
    learning_rate_init=0.00025,
    verbose=1
)
```

The model was trained using an 80/20 train-test split with:

```python
random_state=10
```

---

## 6. Results

The final model achieved the following performance on the held-out test set:

| Metric   |    Value |
| -------- | -------: |
| MSE      | 193.1011 |
| RMSE     |  13.8961 |
| MAE      |   9.0135 |
| R² Score |   0.8465 |

### 6.1 Model Comparison

| Model                                |      MSE |    RMSE |     MAE |     R² |
| ------------------------------------ | -------: | ------: | ------: | -----: |
| HistGradientBoostingRegressor        | 234.8760 | 15.3257 | 10.3982 | 0.8133 |
| MLPRegressor, 512 first layer        | 212.6876 | 14.5838 |  9.5677 | 0.8309 |
| Final MLPRegressor, 1024 first layer | 193.1011 | 13.8961 |  9.0135 | 0.8465 |

The final MLP model improved both error-based metrics and R² compared with the histogram gradient boosting baseline and the smaller MLP model.

---

## 7. Computational Cost

This project was designed around local hardware constraints. The main computational issues were:

* Large dense Coulomb matrix representation.
* High memory usage during feature construction.
* Standard PCA failure due to memory and numerical limitations.
* Slow training time for some models, especially random forests and KNN.

Important implementation decisions included:

* Using `float32` instead of `float64`.
* Using `IncrementalPCA` instead of standard PCA.
* Saving the fitted scaler and PCA objects using `joblib`.
* Training final models on PCA-reduced features rather than raw Coulomb features.

An observed timing test showed that a 50,000 molecule MLP run required approximately 2 minutes and 33 seconds for 30 iterations. This suggested that full-dataset MLP training was feasible after PCA compression.

---

## 8. Repository Structure


```text
molecular-energy-prediction/
│
├── README.md
├── requirements.txt
├── Molecular_Energy_Prediction.ipynb
|
├── models/
│   └── mlp1024_512_256_128_64_64_28_64_16.joblib
│
├── artifacts/
│   ├── train_data_pca1400.joblib
│   └── coloumb_pca_pipeline1400.joblib
│
├── figures/
│   └── model_evaluation_plots.png
│
└── report/
    └── final_report.pdf
```


---

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/tokkser/molecular-energy-prediction.git
cd molecular-energy-prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 10. Requirements

The main Python libraries used are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

A basic `requirements.txt` file can be:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
```

---

## 11. How to Run

### 11.1 Download the Dataset

Download the dataset from Kaggle:

[Predict Molecular Properties Dataset](https://www.kaggle.com/datasets/burakhmmtgl/predict-molecular-properties/data)

Place the JSON files inside the `Dataset/` folder.

### 11.2 Generate Coulomb Matrix and PCA Features

Run the preprocessing section of the notebook:

```python
Training_set=coloumbtransform(n_components=1400)
x_train,y_train=Training_set.coloumbmatrix(jsonpath = dataset_paths)
```

This generates:

```text
train_data_pca1400.joblib
coloumb_pca_pipeline1400.joblib
```

### 11.3 Train the Final Model

Run the final model training cell:

```python
mlp=MLPRegressor(
    hidden_layer_sizes=(1024,512,256,128,64,64,28,64,16),
    learning_rate_init=0.00025,
    verbose=1
)

mlp.fit(x_train,y_train)
```

### 11.4 Evaluate the Model

The notebook reports:

* Mean Squared Error
* Root Mean Squared Error
* Mean Absolute Error
* R² Score
* Residual plots
* Predicted vs actual plots
* Loss curve

---

## 12. Saved Artifacts

The project saves the fitted preprocessing pipeline and trained model using `joblib`.

| File                                        | Description                                                                   |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| `train_data_pca1400.joblib`                 | PCA-reduced feature matrix and target values                                  |
| `coloumb_pca_pipeline1400.joblib`           | Fitted scaler, fitted PCA object, maximum atom count, and atom-count metadata |
| `mlp1024_512_256_128_64_64_28_64_16.joblib` | Final trained MLP model                                                       |

The preprocessing pipeline must be reused during inference. New molecules should be transformed using the saved scaler and PCA objects, not refitted.

---

## 13. Inference

For new JSON molecular data, the inference process is:

```text
New JSON file
    ↓
Extract atoms and coordinates
    ↓
Generate Coulomb matrix using saved max_atoms
    ↓
Flatten matrix
    ↓
Apply saved StandardScaler
    ↓
Apply saved IncrementalPCA
    ↓
Predict energy using saved MLP model
```

Important note:

```text
Use transform, not fit_transform, during inference.
```

Refitting the scaler or PCA on new data would make the feature representation incompatible with the trained model.

---

## 14. Key Findings

* Coulomb matrix features contain strong predictive signal for molecular energy prediction.
* Direct use of the raw Coulomb matrix is memory-intensive at this scale.
* `float32` conversion significantly reduces memory usage.
* Standard PCA is not practical for this feature matrix.
* Incremental PCA enables scalable dimensionality reduction.
* Neural network regression performed better than the tested tree-based baseline.
* The final MLP model achieved an R² score of approximately 0.8465.

---

## 15. Limitations

The project has several limitations:

1. The raw Coulomb matrix is sensitive to atom ordering.
2. PCA compression may discard some useful chemical information.
3. Bond information is not explicitly included.
4. The model is trained only on the chemical distribution present in the selected dataset files.
5. Very large molecules beyond the observed maximum of 121 atoms are not directly supported by the saved pipeline.
6. The model predicts force-field-based energy values from the dataset, not quantum mechanical energies.

---

## 16. Future Work

Possible future improvements include:

* Sorted Coulomb matrix representation.
* Bag of Bonds descriptors.
* Molecular fingerprints.
* Graph neural networks.
* SOAP or atom-centered symmetry function descriptors.
* Inclusion of `shapeM` features.
* Hyperparameter optimization for the final neural network.
* GPU-based deep learning implementation.
* Testing on additional PubChem files.

---

## 17. References

1. Halgren, T. A. Merck Molecular Force Field. I. Basis, Form, Scope, Parameterization, and Performance of MMFF94. Journal of Computational Chemistry 17, 490-519, 1996.

2. Halgren, T. A. Merck Molecular Force Field. VI. MMFF94s Option for Energy Minimization Studies. Journal of Computational Chemistry 20, 720-729, 1999.

3. Kim, S.; Bolton, E. E.; Bryant, S. H. PubChem3D: Shape Compatibility Filtering Using Molecular Shape Quadrupoles. Journal of Cheminformatics 3, 25, 2011.

4. Himmetoglu, B. Tree Based Machine Learning Framework for Predicting Ground State Energies of Molecules. Journal of Chemical Physics 145, 134101, 2016.

5. Rupp, M.; Ramakrishnan, R.; von Lilienfeld, O. A. Machine Learning for Quantum Mechanical Properties of Atoms in Molecules. Journal of Physical Chemistry Letters 6, 3309-3313, 2015.

6. Montavon, G.; Rupp, M.; Gobre, V.; Vazquez-Mayagoitia, A.; Hansen, K.; Tkatchenko, A.; Müller, K.-R.; von Lilienfeld, O. A. Machine Learning of Molecular Electronic Properties in Chemical Compound Space. New Journal of Physics 15, 095003, 2013.

7. Hansen, K.; Montavon, G.; Biegler, F.; Fazli, S.; Rupp, M.; Scheffler, M.; von Lilienfeld, O. A.; Tkatchenko, A.; Müller, K.-R. Assessment and Validation of Machine Learning Methods for Predicting Molecular Atomization Energies. Journal of Chemical Theory and Computation 9, 3543-3556, 2013.

---

## 18. Author

**Abdullah Imran**

---

## 19. Acknowledgement

The dataset used in this project was obtained from Kaggle and is based on molecular data scraped from the PubChem database. The project uses the dataset for educational and research-style machine learning experimentation.

---

## 20. License

This repository is intended for educational and research purposes. Check the original Kaggle dataset page for dataset-specific usage conditions.
