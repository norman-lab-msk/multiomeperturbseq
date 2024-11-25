from sklearn.decomposition._dict_learning import dict_learning
from sklearn.decomposition import SparsePCA
import numpy as np

class NonNegativeSparsePCA(SparsePCA):
    def _fit(self, X, n_components, random_state):
        """Specialized 'fit' for Non-Negative SparsePCA."""

        code_init = self.V_init.T if self.V_init is not None else None
        dict_init = self.U_init.T if self.U_init is not None else None

        # Dictionary learning algorithm with non-negative dictionary atoms
        code, dictionary, E, self.n_iter_ = dict_learning(
            X.T,
            n_components,
            alpha=self.alpha,
            tol=self.tol,
            max_iter=self.max_iter,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=random_state,
            code_init=code_init,
            dict_init=dict_init,
            return_n_iter=True,
            positive_code=True,
        )

        self.components_ = code.T

        # Normalize components
        components_norm = np.linalg.norm(self.components_, axis=1)[:, np.newaxis]
        components_norm[components_norm == 0] = 1
        self.components_ /= components_norm
        self.n_components_ = len(self.components_)

        self.error_ = E
        return self