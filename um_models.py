import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator

from util import auuc_sep_rel_prop1

class TMEstimator(BaseEstimator):
    def __init__(self, C=1e1, solver='lbfgs', penalty='l2', class_weight=None):
        self.C = C
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight  

    def fit(self, x, yt):
        y, t = yt[:,0], yt[:,1]
        x_t = x[t==1]
        y_t = y[t==1]
        x_c = x[t==0]
        y_c = y[t==0]
        
        self._model_t = LogisticRegression(
            C=self.C, 
            solver=self.solver, 
            penalty=self.penalty, 
            class_weight=self.class_weight,
            random_state=0
        )
        self._model_t.fit(x_t, y_t)
        
        self._model_c = LogisticRegression(
            C=self.C, 
            solver=self.solver, 
            penalty=self.penalty, 
            class_weight=self.class_weight,
            random_state=0
        )
        self._model_c.fit(x_c, y_c)
        return self

    def score(self, x, yt, top_ratio=1):
        y, t = yt[:,0], yt[:,1]
        u = self.predict_uplift(x)
        return auuc_sep_rel_prop1(y, t, u)

    def predict_uplift(self, x):
        return self._model_t.predict_proba(x)[:, 1] - self._model_c.predict_proba(x)[:, 1]
    

class CVTEstimator(BaseEstimator):
    def __init__(self, C=1e1, solver='lbfgs', penalty='l2', class_weight=None):
        self.C = C
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight 
        

    def fit(self, x, yt):
        y, t = yt[:,0], yt[:,1]
        z = np.array(y == t, dtype=int)
        
        self._model_ = LogisticRegression(
            C=self.C, 
            solver=self.solver, 
            penalty=self.penalty, 
            class_weight=self.class_weight,
            random_state=0
        )

        self._model_.fit(x, z)
        return self

    def score(self, x, yt, top_ratio=1):
        y, t = yt[:,0], yt[:,1]
        u = self.predict_uplift(x)
        return auuc_sep_rel_prop1(y, t, u)

    def predict_proba(self, x):
        return self._model_.predict_proba(x)

    def predict_uplift(self, x):
        # Definition: p(z=1|x) = (p_T(y=1|x) - p_C(y=1|x)) / 2 + 1 / 2
        p_z = self.predict_proba(x)[:, 1]
        return 2. * p_z - 1
    
    
class MOMEstimator(BaseEstimator):
    def __init__(self, alpha=1e0, solver='auto', normalize=False):
        self.alpha = alpha
        self.normalize = normalize
        self.solver = solver

    def fit(self, x, yt):
        y, t = yt[:,0], yt[:,1]
        e = np.mean(t)
        z = y * (t - e) / (e*(1-e))
        
        self._model_ = Ridge(
            alpha=self.alpha, 
            solver=self.solver, 
            normalize=self.normalize,
            random_state=0
        )
        self._model_.fit(x, z)
        return self

    def score(self, x, yt, top_ratio=1):
        y, t = yt[:,0], yt[:,1]
        u = self.predict_uplift(x)
        return auuc_sep_rel_prop1(y, t, u)

    def predict(self, x):
        return self._model_.predict(x)

    def predict_uplift(self, x):
        p_z = self.predict(x)
        return p_z
    
    
class SDREstimator(BaseEstimator):
    def __init__(self, C=1e1, solver='lbfgs', penalty='l2', class_weight=None, reg=1e0, sparse=False):
        self.C = C
        self.solver = solver
        self.penalty = penalty
        self.class_weight = class_weight
        self.reg = reg
        self.sparse = sparse
        

    def fit(self, x, yt):
        y, t = yt[:,0], yt[:,1]
        if sparse:
            x_full = sparse.vstack((
                sparse.hstack((x[t==1], self.reg*x[t==1], np.zeros((x[t==1].shape))), format='csr'),
                sparse.hstack((x[t==0], np.zeros((x[t==0].shape)), self.reg*x[t==0]), format='csr')
            ))
        else:
            x_full = np.vstack((
                np.hstack((x[t==1], self.reg*x[t==1], np.zeros((x[t==1].shape)))),
                np.hstack((x[t==0], np.zeros((x[t==0].shape)), self.reg*x[t==0]))
            ))
        y_full = np.vstack((
            y[t==1][..., None],
            y[t==0][..., None]
        ))
        
        self._model_ = LogisticRegression(
            C=self.C, 
            solver=self.solver, 
            penalty=self.penalty, 
            class_weight=self.class_weight,
            random_state=0
        )
        self._model_.fit(x_full, y_full)
        return self

    def score(self, x, yt, top_ratio=1):
        y, t = yt[:,0], yt[:,1]
        u = self.predict_uplift(x)
        return auuc_sep_rel_prop1(y, t, u)

    def predict_proba(self, x):
        return self._model_.predict_proba(x)

    def predict_uplift(self, x):
        if sparse:
            y_t = self.predict_proba(sparse.hstack((x, x, np.zeros((x.shape))), format='csr'))[:, 1]
            y_c = self.predict_proba(sparse.hstack((x, np.zeros((x.shape)), x), format='csr'))[:, 1]
        else:
            y_t = self.predict_proba(np.hstack((x, x, np.zeros((x.shape)))))[:, 1]
            y_c = self.predict_proba(np.hstack((x, np.zeros((x.shape)), x)))[:, 1]
        return y_t - y_c