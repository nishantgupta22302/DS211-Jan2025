# DS211-Jan25-CDS
# Nishant Gupta SR. 22302
# Lecture 02

import numpy as np
import pandas as pd

# Load dataset using pandas
df = pd.read_csv("real_estate_dataset.csv")

# Get and print number of samples and features
n_samples,n_features = df.shape
print(f"(Number of Samples, Number of Features):{df.shape}")

# Get and save column names
col = df.columns
np.savetxt("column_name.txt", col, fmt="%s")

# Include features: Square_Feet, Garage_Size, Location_Score, Distance_to_Center
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]
# Set target: price
y = df['Price']

# Update/print X shape/datatype
n_samples,n_features = X.shape
print(f"Shape of X:{X.shape}")
print(f"Datatype of X:\n{X.dtypes}")

# Develop linear model for 4-X features (+1 for bias)
coeffs = np.ones(n_features+1)
predictions_by_defn = X@coeffs[1:] + coeffs[0]
# Concat a column vector of ones in front of X for bias term
X = np.hstack((np.ones((n_samples,1)),X))
predictions = X@coeffs
is_same = np.allclose(predictions_by_defn,predictions)
print(f"Are predictions same as predictions_by_def? {is_same}")

# Calculate errors
errors = y - predictions
# Calculate relative errors
rel_errors = errors/y

# Calculate mean squares of errors
loss_loop = 0
for i in errors:
  loss_loop += i**2
loss_loop /= len(errors)
# Calculate mean of squares of error using norm (errors)^T(errors)
loss_matrix = (errors.T @ errors)/len(errors)
# Compare the two methods
is_diff = np.isclose(loss_loop,loss_matrix)
print(f"Are losses same? {is_diff}")

# Print errors
print(f"Length of errors: {errors.size}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")
print(f"L2 norm of relative errors: {np.linalg.norm(rel_errors)}")

# Optimizatiion? To minimize mean squared error b/w predicted values
# and actual values. This is Least Squares Problem.
# Solution? Search for coefficients for which gradient of objective
# function is zero.
loss_matrix = (1/n_samples) * (y-X@coeffs).T @ (y-X@coeffs)

fgrad_matrix = -(2/n_samples)*X.T@(y-X@coeffs)
# XT.(y-Xc)=0 => C = inv(XT.X)XTy
coeffs = np.linalg.inv(X.T@X)@X.T@y
# Save coefficients
np.savetxt("coeffs.txt", coeffs, delimiter=",")

prediction_model = X@coeffs
errors_model = y - prediction_model

rel_errors_model = errors_model/y

print(f"rel_errors_model L2 norm, {np.linalg.norm(rel_errors_model, ord=2)}")


# -----------------Using all features-----------------
# Remove target column: Price
X = df.drop(columns=['Price'])

# Concat a column vector of ones in front of x for bias term
X = np.hstack((np.ones((n_samples,1)),X))

# Inverting grad_matrix for estimating coefficients
coeffs_all = np.linalg.inv(X.T@X)@X.T@y

# Computing predicted values
prediction_all = X@coeffs_all

# Computing errors and relative errors
errors_all = y - prediction_all
rel_errors_all = errors_all/y

# Computing/Printing L2 norm of relative errors
print(f"L2 norm of relative errors: {np.linalg.norm(rel_errors_all, ord=2)}")
np.savetxt("coeffs_all.txt", coeffs_all, delimiter=",")

# Save coefficients
np.savetxt("coeffs_all.txt", coeffs_all, delimiter=",")


# ----------Rank of XT.X----------
print(f"Rank(X.T@X): {np.linalg.matrix_rank(X.T@X)}")


# -----------------Solve by QR factorization-----------------
# XT.(y-X.c)=0 => XT.y = XT.X.c
# Let x = QR   => RT.QT.y = RT.QT.Q.R.c = RT.R.c
#              => QT.y = R.c = b
#              => c = inv(R).b (R is lower triangular)
Q,R = np.linalg.qr(X)

# ----------Print shapes of Q,R----------
print(f"Shape(Q): {Q.shape}")
print(f"Shape(R): {R.shape}")

# -----------Check if QT.Q=I-------------
I = Q.T@Q
np.savetxt("I.csv", I, delimiter=",")

b = Q.T@y
print(f"Shape(b): {b.shape}")
print(f"Shape(R): {R.shape}")
n_samples,n_features = X.shape

coeffs_qr_loop = np.zeros(n_features)

# Compute Coefficients by back substitution
for i in range (n_features-1,-1,-1):
  coeffs_qr_loop[i]=b[i]
  for j in range(i+1, n_features):
    coeffs_qr_loop[i] -= R[i,j]*coeffs_qr_loop[j]
  coeffs_qr_loop[i] /=R[i,i]

# Save Coefficients
np.savetxt("coeffs_qr.txt", coeffs_qr_loop, delimiter=",")

# ----------------------Solve using SVD----------------------
# XT.(y-Xc)=0 => XT.y = XT.X.c                 (U.UT=I, V.VT)
# Let X = U.S.VT  => V.ST.UT.y = V.ST.UT.U.S.VT.c
#                    V.ST.UT.y = V.ST.S.VT.c
#                    V.inv(S).UT.y = c
# => c = (VT.T).(inv_S).(UT).y
# SVD
U, S, VT = np.linalg.svd(X, full_matrices=False)
# S is diagonal
inv_S = np.diag(1/S)
# Compute Coefficients
coeffs_svd = (VT.T) @ (inv_S) @ (U.T) @ y
# Save Coefficients
np.savetxt("coeffs_svd.txt", coeffs_svd, delimiter=",")

# ------------Solve using EigenValue Decomposition------------
# XT.(y-Xc)=0 => XT.y = XT.X.c                 (S.ST=I)
# Let XT.X = S.D.ST   => XT.y = S.D.ST.c
#                     => ST.XT.y = D.ST.c
#                     => inv(D).ST.XT.y = ST.c
#                     => S.inv(D).ST.XT.y = c
# => c = S.inv(D).ST.XT.y
# EVD
D, S = np.linalg.eigh(X.T @ X)
# D is diagonal
inv_D = np.diag(1/D)
# Compute Coefficients
coeffs_evd = (S) @ (inv_D) @ (S.T) @ (X.T) @ y
# Save Coefficients
np.savetxt("coeffs_evd.txt", coeffs_evd, delimiter=",")
