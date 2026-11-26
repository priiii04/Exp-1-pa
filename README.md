# Experiment 1: Exploring NumPy and Pandas

## Aim

To understand the use of NumPy and Pandas for performing basic data
manipulation and analysis in Python.

## Algorithm

1.  Import the NumPy library for matrix and numerical operations.
2.  Define a two-dimensional square matrix `A` and a vector `b`.
3.  Compute the transpose of matrix `A` to interchange rows and columns.
4.  Calculate the inverse of matrix `A` using `np.linalg.inv(A)` (ensure
    `A` is non-singular).
5.  Solve the linear equation `A × x = b` using `np.linalg.solve(A, b)`.
6.  Perform eigen decomposition to find eigenvalues and eigenvectors
    using `np.linalg.eig(A)`.
7.  Display the results for the matrix, transpose, inverse, solution
    vector, and eigenvalues.
8.  End the program.

## Formulas / Theory

-   **Transpose:** `Aᵀ` --- interchange rows and columns.
-   **Inverse:** `A⁻¹ = adj(A) / |A|`, provided `det(A) ≠ 0`.
-   **Linear system:** `A × x = b  ⇒  x = A⁻¹b`.
-   **Eigen decomposition:** `A v = λ v`, where `λ` is an eigenvalue and
    `v` is an eigenvector.

## Coding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data={ 'Advertising':[20,25,30,35,40,45,50,55],'Price':[200,190,180,170,160,150,140,130],
      'Customer_Rating':[6,7,7,8,8,9,9,10],'Sales':[180,200,220,240,260,285,30,330]}
df=pd.DataFrame(data)
print("-----Dataset-----")
print(df)

X=df[['Advertising','Price','Customer_Rating']]
Y=df['Sales']

model=LinearRegression()
model.fit(X,Y)

print("\n Intercept(a):",model.intercept_)
print("Coefficients (b1,b2,b3):",model.coef_)

print(f"\n Regression Equation:Sales={model.intercept_:.2f}+{model.coef_[0]:.2f}xAdvertising+{model.coef_[1]:.2f}xPrice+{model.coef_[2]:.2f}xCustomer_Rating")


## Output


## Result

