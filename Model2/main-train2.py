import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

# Ruta del model
MODEL_PATH = os.environ["MODEL_PATH"]

# Dades de mostra
np.random.seed(2)
x = np.random.normal(3, 1, 100).reshape(-1, 1)
y = np.random.normal(150, 40, 100) / x.ravel()

# Divisió en conjunts d'entrenament i test
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

# Creació i entrenament del model amb un arbre de decisió regressiu
model = DecisionTreeRegressor(max_depth=4)
model.fit(train_x, train_y)
print("Model entrenat amb èxit")

# Avaluació del model
r2 = r2_score(test_y, model.predict(test_x))
print("Model Score:", r2)

# Guardar el model
np.save(MODEL_PATH, model)
