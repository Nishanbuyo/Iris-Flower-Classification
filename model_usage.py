from joblib import dump, load
import numpy as np
model=load("iris_model.joblib")
features=np.array([[4.7, 3.2, 1.3, 0.2 ]])
print(model.predict(features))