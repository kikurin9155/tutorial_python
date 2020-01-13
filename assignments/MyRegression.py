import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

class MyRegression(BaseEstimator,RegressorMixin):
	def fit(self, X, y):
          self.coef_ = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y)) 
        
          return self

        def predict(self,X):
          y = np.dot(X, self.coef_)
        
          return y

if __name__=="__main__":
	clf = MyRegression()
