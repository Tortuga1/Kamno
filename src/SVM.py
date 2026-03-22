from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

df = pd.read_csv("data/input_1.csv")
df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["mol"].notnull()].reset_index(drop=True)

y = df["Inh Power"].to_numpy()

#morgan fingerprints
fing_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X_fp = np.array(fing_gen.GetFingerprints(list(df["mol"])))

#RDKit Descriptors
desc_dicts = [Descriptors.CalcMolDescriptors(mol) for mol in df["mol"]]
desc_df = pd.DataFrame(desc_dicts)
desc_df = desc_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
X_desc = desc_df.values.astype(np.float64)

#print(f"X_fp: {X_fp.shape}, X_desc: {X_desc.shape}, y: {y.shape}")

#split data
idx = np.arange(len(y))
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=10)

X_fp_train, X_fp_test = X_fp[idx_train], X_fp[idx_test]
X_desc_train, X_desc_test = X_desc[idx_train], X_desc[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

#SVR on Morgan fingeprints

print("=" * 55)
print("SVR on Morgan Fingerprints")
print("=" * 55)

pipe_fp = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR()),
])

param_grid_fp = {
    "svr__kernel":  ["rbf", "linear"], #radial basis function creates a complex boundary not a linear 
    "svr__C":       [0.1, 1, 10, 100], #flatness of cylinder and tolerating errors (difference between this an svr__epsilon?)
    "svr__epsilon": [0.1, 0.5, 1.0], #acceptable error (width of the cylinder) - how to change this value for larger dataset?
    "svr__gamma":   ["scale", "auto"],  
}

gs_fp = GridSearchCV(
    pipe_fp, param_grid_fp,
    cv=5, scoring="r2",
    n_jobs=-1, verbose=3,
)
gs_fp.fit(X_fp_train, y_train)

y_pred_fp = gs_fp.predict(X_fp_test)
r2_fp  = r2_score(y_test, y_pred_fp)
mse_fp = mean_squared_error(y_test, y_pred_fp)
rmse_fp = np.sqrt(mse_fp)

plt.scatter(y_test, y_pred_fp)
plt.grid()
plt.show()

print(f"\nBest params : {gs_fp.best_params_}")
print(f"R²  (test)  : {r2_fp:.4f}")
print(f"RMSE (test) : {rmse_fp:.4f}\n")

#SVR on RDKit Descriptors -need scaling
print("=" * 55)
print("SVR on RDKit Descriptors")
print("=" * 55)

pipe_desc = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR()),
])

param_grid_desc = {
    "svr__kernel":  ["rbf", "linear"],
    "svr__C":       [0.1, 1, 10, 100],
    "svr__epsilon": [0.1, 0.5, 1.0],
    "svr__gamma":   ["scale", "auto"],
}

gs_desc = GridSearchCV(
    pipe_desc, param_grid_desc,
    cv=5, scoring="r2",
    n_jobs=-1, verbose=3,
)
gs_desc.fit(X_desc_train, y_train)

y_pred_desc = gs_desc.predict(X_desc_test)
r2_desc  = r2_score(y_test, y_pred_desc)
mse_desc = mean_squared_error(y_test, y_pred_desc)
rmse_desc = np.sqrt(mse_desc)

print(f"\nBest params : {gs_desc.best_params_}")
print(f"R²  (test)  : {r2_desc:.4f}")
print(f"RMSE (test) : {rmse_desc:.4f}\n")

plt.scatter(y_test, y_pred_desc)
plt.grid()
plt.show()

#comparison
print("=" * 55)
print("Comparison")
print("=" * 55)
results = pd.DataFrame({
    "Model":      ["SVR — Morgan Fingerprints", "SVR — RDKit Descriptors"],
    "Best kernel": [gs_fp.best_params_["svr__kernel"],
                    gs_desc.best_params_["svr__kernel"]],
    "Best C":     [gs_fp.best_params_["svr__C"],
                   gs_desc.best_params_["svr__C"]],
    "R² (test)":  [round(r2_fp, 4),   round(r2_desc, 4)],
    "RMSE (test)":[round(rmse_fp, 4), round(rmse_desc, 4)],
})
print(results.to_string(index=False))


#using this code to predict a molecule