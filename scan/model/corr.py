"""
Module for estimating the relationship between functional MRI signals
and physio signals at successive temporal lags of the physio signal
"""
from typing import List, Literal, Tuple

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

from patsy import dmatrix

from scan.io.write import DistributedLagModelPredResults


class BSplineLagBasis(BaseEstimator, TransformerMixin):
    """
    Spline basis for modeling temporal lags of a physio signal based on
    scikit-learn fit/transform API. Specifically, a B-spline basis is fit
    along the columns of a lag matrix (rows: time courses; columns: lags),
    where the first column is the original time course, the second column
    is the original time course lagged by one time point, the third column
    lagged by two time points, out to N lags (specified by nlags parameter).
    You can also specify negative lags (specified by neg_nlags parameter).

    Attributes
    ----------
    nlags: int
        number of lags (shifts) of the signal in the forward direction
    nlags_neg: int
        number of lags (shifts) of the signal in the negative direction. 
        Must be a negative integer. This allows modeling the association between
        functional and physio signals where the functional leads the physio signal.
    n_knots: int
        number of knots in the spline basis across temporal lags. Controls
        the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots: List[int]
        Locations of knots in spline basis across temporal lags. If provided,
        the n_knots parameter is ignored.
    basis: Literal['ns','bs']
        basis type for the spline basis. 'ns' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        fit B-spline basis to lags of the signal.
    transform(X, y)
        project lags of the signal onto the B-spline basis. X is the physio
        time course represented in an ndarray with time points along the
        rows and a single column (# of time points, 1).

    """
    def __init__(
        self, 
        nlags: int, 
        neg_nlags: int = 0, 
        n_knots: int = 5,
        knots: List[int] = None,
        basis: Literal['ns','bs'] = 'bs'
    ):
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        
        # specify array of lags
        self.lags = np.arange(neg_nlags, nlags+1)
        # specify knots parameters
        self.n_knots = n_knots
        self.knots = knots
        self.basis = basis

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        create spline basis over lags of physio signal

        Parameters
        ----------
        X: np.ndarray
            The physio time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        y: None
            Not used, for consistency with sklearn API
        """
        # create spline basis from sklearn SplineTransformer
        if self.knots is not None:
            self.basis = dmatrix(
                f'{self.basis}(x, knots=self.knots) - 1',
                {'x': self.lags}
            )
        else:
            self.basis = dmatrix(
                f'{self.basis}(x, df=self.n_knots) - 1',
                {'x': self.lags}
            )
    
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        project lags of physio signal onto spline basis

        Parameters
        ----------
        X: np.ndarray
            The physio time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        y: None
            Not used, for consistency with sklearn API

        Returns
        -------
        lag_proj: np.ndarray
            Physio signal projected on B-spline basis.
        """
        # create lag matrix
        lagmat = _lag_mat(X, self.lags)
        # get number of splines
        n_splines = self.basis.shape[1]
        # allocate memory
        lag_proj = np.empty(
            (lagmat.shape[0], n_splines),
            dtype=lagmat.dtype
        )
        for l in np.arange(n_splines):
            lag_proj[:,l] = np.dot(lagmat, self.basis[:,l])

        return lag_proj


class DistributedLagModel:
    """
    Distributed lag model of physio signals regressed onto functional
    MRI signals at each voxel (mass-univariate). Specifically, lags of
    the physio signal are projected on a B-spline basis and regressed onto
    functional MRI signals.

    nlags: int
        number of lags (shifts) of the physio signal in the forward direction
    nlags_neg: int
        number of lags (shifts) of the physio signal in the negative direction.
        Must be a negative integer. This allows modeling the association between
        functional and physio signals where the functional leads the physio signal.
    n_knots: int
        number of knots in the spline basis across temporal lags. Controls
        the temporal resolution of the basis, such that more knots results
        in the ability to capture more complex curves (at the expense of
        potential overfitting) (default: 5)
    knots: List[int]
        knot locations for the spline basis across temporal lags. If supplied, this
        overrides the n_knots parameter.
    alpha: float
        regularization strength of the Ridge regression [0, inf]. Greater values
        results in greater regularization (default: 0.01).
    basis: Literal['cr','bs']
        basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.

    Methods
    -------
    fit(X,y):
        regress lags of physio signal onto voxel-wise functional time courses.

    predict()

    """
    def __init__(
        self,
        nlags: int,
        neg_nlags: int = 0,
        n_knots: int = 5,
        knots: List[int] = None,
        alpha: float = 0.01,
        basis: Literal['cr','bs'] = 'bs'
    ):
        # specify array of lags
        self.nlags = nlags
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        self.neg_nlags = neg_nlags
        self.n_knots = n_knots
        self.knots = knots
        self.alpha = alpha
        self.basis_type = basis

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray = None) -> None:
        """
        fit regression model of physio lag spline basis regressed on functional
        time courses

        Parameters
        ----------
        X: np.ndarray
            The physio time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        Y: np.ndarray
            functional MRI time courses represented in an ndarray with time
            points along the rows and vertices in the columns (# of time
            points, # of vertices).
        weights: np.ndarray
            weights for each time point. If None, all weights are set to 1.
        """
        # create B-spline basis across lags of physio signal
        self.basis = BSplineLagBasis(
            nlags=self.nlags, neg_nlags=self.neg_nlags,
            n_knots=self.n_knots, knots=self.knots, basis=self.basis_type
        )
        self.basis.fit(X)
        # project physio signal lags on B-spline basis
        x_basis = self.basis.transform(X)
        # create nan mask for x_basis
        self.nan_mask = np.isnan(x_basis).any(axis=1)
        # if weights is None, set to ones
        if weights is None:
            weights = np.ones(X.shape[0])
        # fit Ridge regression model
        self.glm = Ridge(alpha=self.alpha, fit_intercept=False)
        self.glm.fit(
            x_basis[~self.nan_mask], 
            Y[~self.nan_mask], 
            sample_weight=weights[~self.nan_mask]
        )
        return self

    def predict(
        self,
        lag_max: float = None,
        lag_min: float = None,
        n_eval: int = 30,
        pred_val: float = 1.0
    ) -> DistributedLagModelPredResults:
        """
        Get predicted functional time course at lags of the physio signal.

        Parameters
        ----------
        lag_max: float
            The length of lags of the physio signal to predict functional time
            courses for. If None, set to nlag specified in initialization. (
            default: None)
        lag_min: float
            The minimium lag of the physio signal to predict functional time
            courses for. Must be a negative integer. If None, set to neg_nlag
            specified in initialization. (default: None)
        n_eval: int
            Number of interpolated samples to predict functional time
            courses for between lag_min and lag_max.
        pred_val: float
            The predicted physio signal value used to predict functional time
            courses (default: 1.0).

        Returns
        -------
        dlm_pred: DistributedLagModelPredResults
            Container object for distribued lag model prediction results
        """
        # if lag_max is None, set nlags
        if lag_max is None:
            lag_max = self.nlags
        # if lag_min is None, set neg_nlags
        if lag_min is None:
            lag_min = self.neg_nlags
        else:
            if lag_min > 0:
                raise ValueError("lag_min must be a negative integer")

        # specify lags for prediction (number of samples set by n_eval )
        pred_lags = np.linspace(lag_min, lag_max, n_eval)
        # project lag vector onto B-spline basis
        pred_basis = dmatrix(
            self.basis.basis.design_info,
            {'x': pred_lags.reshape(-1, 1)}
        )
        # project prediction value on lag B-spline basis
        physio_pred = [
            pred_val * pred_basis[:, l]
            for l in range(pred_basis.shape[1])
         ]
        physio_pred = np.vstack(physio_pred).T
        # Get predictions from model
        pred_func = self.glm.predict(physio_pred)
        # package output in container object
        dlm_pred = DistributedLagModelPredResults(
            pred_func = pred_func,
            dlm_params = {
                'lag_max': lag_max,
                'lag_min': lag_min,
                'n_eval': n_eval,
                'pred_lags': pred_lags,
                'basis_type': self.basis_type,
            }
        )
        return dlm_pred


class DistributedLagNonLinearModel:
    """
    Distributed lag non-linear model that forms a tensor product basis over both
    the predictor variable and lag dimensions. This allows modeling non-linear
    relationships in both dimensions simultaneously.
    
    Parameters
    ----------
    nlags : int
        Number of lags (shifts) of the physio signal in the forward direction
    neg_nlags : int
        Number of lags (shifts) of the physio signal in the negative direction.
        Must be a negative integer. This allows modeling the association between
        functional and physio signals where the functional leads the physio signal.
    n_knots_lag : int
        Number of knots in the spline basis across temporal lags
    n_knots_var : int
        Number of knots in the spline basis across the predictor variable
    knots_lag : list[int]
        Locations of knots in the spline basis across temporal lags. If provided,
        the n_knots_lag parameter is ignored.
    knots_var : list[int]
        Locations of knots in the spline basis across the predictor variable. If provided,
        the n_knots_var parameter is ignored.
    basis_var : Literal['cr','bs']
        basis type for the spline basis across the predictor variable. 'cr' for natural spline, 'bs' for B-spline.
    basis_lag : Literal['cr','bs']
        basis type for the spline basis across temporal lags. 'cr' for natural spline, 'bs' for B-spline.
    alpha : float
        Regularization strength of the Ridge regression [0, inf] (default: 0.01)
    """
    def __init__(
        self,
        nlags: int,
        neg_nlags: int = 0,
        n_knots_lag: int = 5,
        n_knots_var: int = 5,
        knots_lag: List[int] = None,
        knots_var: List[int] = None,
        basis_var: Literal['cr','bs'] = 'bs',
        basis_lag: Literal['cr','bs'] = 'bs',
        alpha: float = 0.01,
    ):
        self.nlags = nlags
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        self.neg_nlags = neg_nlags

        # get knots parameters
        self.n_knots_lag = n_knots_lag
        self.n_knots_var = n_knots_var
        self.knots_lag = knots_lag
        self.knots_var = knots_var
        # specify basis types
        self.basis_lag_type = basis_lag
        self.basis_var_type = basis_var
        # set regularization strength
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray = None) -> None:
        """
        Fit regression model using tensor product basis over both predictor
        and lag dimensions.
        
        Parameters
        ----------
        X : np.ndarray
            The physio time course with shape (n_timepoints, 1)
        Y : np.ndarray
            Functional MRI time courses with shape (n_timepoints, n_vertices)
        weights : np.ndarray
            Weights for each time point. If None, all weights are set to 1.
        """
        # Create basis for predictor variable
        if self.knots_var is not None:
            self.var_basis = dmatrix(
                f'{self.basis_var_type}(x, knots=self.knots_var) - 1',
                {'x': X}
            )
        else:
            self.var_basis = dmatrix(
                f'{self.basis_var_type}(x, df=self.n_knots_var) - 1',
                {'x': X}
            )

        # Create basis for lags using BSplineLagBasis
        self.lag_basis = BSplineLagBasis(
            nlags=self.nlags,
            neg_nlags=self.neg_nlags,
            n_knots=self.n_knots_lag,
            knots=self.knots_lag,
            basis=self.basis_lag_type
        )
        self.lag_basis.fit(X)
        lag_basis = self.lag_basis.basis

        # Create tensor product basis
        self.tensor_basis, self.nan_mask = self._create_tensor_basis(
            self.var_basis, lag_basis
        )

        # if weights is None, set to ones
        if weights is None:
            weights = np.ones(X.shape[0])

        # Fit Ridge regression
        self.glm = Ridge(alpha=self.alpha, fit_intercept=False)
        self.glm.fit(self.tensor_basis[~self.nan_mask], Y[~self.nan_mask])
        
        return self

    def _create_tensor_basis(
        self, 
        var_basis: np.ndarray, 
        lag_basis: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tensor product basis by taking outer product of variable and lag bases. 
        Also returns nans mask for tensor_basis to remove nans introduced by lags at the 
        beginning of the tensor_basis.

        Parameters
        ----------
        var_basis : np.ndarray
            Basis functions for predictor variable
        lag_basis : np.ndarray
            Basis functions for lags
            
        Returns
        -------
        np.ndarray
            Tensor product basis matrix
        """
        n_samples = var_basis.shape[0]
        n_var_basis = var_basis.shape[1]
        n_lag_basis = lag_basis.shape[1]
        
        tensor_basis = np.zeros((n_samples, n_var_basis * n_lag_basis))
        for i in range(n_var_basis):
            var_lag_mat = _lag_mat(var_basis[:, [i]], self.lag_basis.lags)
            for j in range(n_lag_basis):
                tensor_basis[:, i * n_lag_basis + j] = np.dot(var_lag_mat, lag_basis[:, j])
        
        # lags introduce nans at the beginning of the tensor_basis
        nan_mask = np.isnan(tensor_basis).any(axis=1)

        return tensor_basis, nan_mask

    def predict(
        self,
        lag_max: float = None,
        lag_min: float = None,
        n_eval: int = 30,
        pred_val: float = 1.0
    ) -> DistributedLagModelPredResults:
        """
        Get predicted functional time course at lags of the physio signal.
        
        Parameters
        ----------
        lag_max : float
            Maximum lag for prediction
        lag_min : float
            Minimum lag for prediction. Must be a negative integer.
        n_eval : int
            Number of evaluation points
        pred_val : float
            Value of predictor variable for prediction
            
        Returns
        -------
        DistributedLagModelPredResults
            Container with prediction results
        """
        if lag_max is None:
            lag_max = self.nlags
        if lag_min is None:
            lag_min = self.neg_nlags
        else:
            if lag_min > 0:
                raise ValueError("lag_min must be a negative integer")
            
        # Create evaluation points
        pred_lags = np.linspace(lag_min, lag_max, n_eval)
        
        # Transform predictor value - create array with n_eval rows
        pred_val_array = np.full((n_eval, 1), pred_val)
        pred_val_basis = dmatrix(
            self.var_basis.design_info,
            {'x': pred_val_array}
        )

        # Transform lags using BSplineLagBasis
        pred_lag_basis = dmatrix(
            self.lag_basis.basis.design_info,
            {'x': pred_lags}
        )

        # Create tensor product basis for prediction
        # pred_basis = self._create_tensor_basis(pred_val_basis, pred_lag_basis)
        pred_basis = [
            pred_val_basis[:,[v]]*pred_lag_basis[:,[l]] 
            for v in range(pred_val_basis.shape[1]) 
            for l in range(pred_lag_basis.shape[1])
        ]
        pred_basis = np.hstack(pred_basis)

        # Get predictions
        pred_func = self.glm.predict(pred_basis)
        
        # Package results
        dlm_pred = DistributedLagModelPredResults(
            pred_func=pred_func,
            dlm_params={
                'lag_max': lag_max,
                'lag_min': lag_min,
                'n_eval': n_eval,
                'pred_lags': pred_lags,
                'basis_var_type': self.basis_var_type,
                'basis_lag_type': self.basis_lag_type
            }
        )
        
        return dlm_pred


class MultivariateDistributedLagModel:
    """
    Multivariate distributed lag model that extends DistributedLagModel to handle
    multiple time series and their interactions. Creates a tensor product basis
    across all time series and their lags.

    Parameters
    ----------
    nlags : int
        Number of lags (shifts) of each physio signal in the forward direction
    neg_nlags : int
        Number of lags (shifts) of each physio signal in the negative direction.
        Must be a negative integer.
    n_knots : int
        Number of knots in the spline basis across temporal lags for each signal
    knots : List[int]
        List of knot locations for each signal's spline basis. If provided,
        overrides n_knots parameter.
    alpha : float
        Regularization strength of the Ridge regression [0, inf] (default: 0.01)
    basis : Literal['cr','bs']
        Basis type for the spline basis. 'cr' for natural spline, 'bs' for B-spline.
    """
    def __init__(
        self,
        nlags: int,
        neg_nlags: int = 0,
        n_knots: int = 5,
        knots: List[int] = None,
        alpha: float = 0.01,
        basis: Literal['cr','bs'] = 'bs'
    ):
        self.nlags = nlags
        if neg_nlags > 0:
            raise ValueError("neg_nlags must be a negative integer")
        self.neg_nlags = neg_nlags
        self.n_knots = n_knots
        self.knots = knots
        self.alpha = alpha
        self.basis_type = basis
        self.n_signals = None  # Will be set during fit

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray = None) -> None:
        """
        Fit regression model using tensor product basis across all signals and their lags.

        Parameters
        ----------
        X : np.ndarray
            The physio time courses with shape (n_timepoints, n_signals)
        Y : np.ndarray
            Functional MRI time courses with shape (n_timepoints, n_vertices)
        weights : np.ndarray
            Weights for each time point. If None, all weights are set to 1.
        """
        self.n_signals = X.shape[1]
        
        # Create basis for each signal
        self.bases = []
        for i in range(self.n_signals):
            basis = BSplineLagBasis(
                nlags=self.nlags,
                neg_nlags=self.neg_nlags,
                n_knots=self.n_knots,
                knots=self.knots if self.knots is not None else None,
                basis=self.basis_type
            )
            basis.fit(X[:, [i]])
            self.bases.append(basis)

        # Create tensor product basis across all signals
        self.tensor_basis, self.nan_mask = self._create_tensor_basis(X)

        # if weights is None, set to ones
        if weights is None:
            weights = np.ones(X.shape[0])

        # Fit Ridge regression
        self.glm = Ridge(alpha=self.alpha, fit_intercept=False)
        self.glm.fit(
            self.tensor_basis[~self.nan_mask], 
            Y[~self.nan_mask], 
            sample_weight=weights[~self.nan_mask]
        )
        
        return self

    def _create_tensor_basis(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create tensor product basis by taking outer product of all signal bases.
        
        Parameters
        ----------
        X : np.ndarray
            Input signals with shape (n_timepoints, n_signals)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tensor product basis matrix and nan mask
        """
        # Get basis for each signal
        signal_bases = []
        for i, basis in enumerate(self.bases):
            signal_basis = basis.transform(X[:, [i]])
            signal_bases.append(signal_basis)

        # Create tensor product basis
        tensor_basis = np.hstack(signal_bases)
        for i in range(len(signal_bases)):
            for j in range(i+1, len(signal_bases)):
                outer_product = self._outer_product(signal_bases[i], signal_bases[j])
                tensor_basis = np.hstack([tensor_basis, outer_product])

        # Create nan mask
        nan_mask = np.isnan(tensor_basis).any(axis=1)

        return tensor_basis, nan_mask

    def _outer_product(self, basis1: np.ndarray, basis2: np.ndarray) -> np.ndarray:
        """
        Compute outer product of two basis matrices.
        
        Parameters
        ----------
        basis1 : np.ndarray
            First basis matrix
        basis2 : np.ndarray
            Second basis matrix
            
        Returns
        -------
        np.ndarray
            Outer product basis matrix
        """
        n_samples = basis1.shape[0]
        n_basis1 = basis1.shape[1]
        n_basis2 = basis2.shape[1]
        
        outer_basis = np.zeros((n_samples, n_basis1 * n_basis2))
        for i in range(n_basis1):
            for j in range(n_basis2):
                outer_basis[:, i * n_basis2 + j] = basis1[:, i] * basis2[:, j]
                
        return outer_basis

    def predict(
        self,
        lag_max: float = None,
        lag_min: float = None,
        n_eval: int = 30,
        pred_vals: List[float] = None
    ) -> DistributedLagModelPredResults:
        """
        Get predicted functional time course at specified values for each signal.
        
        Parameters
        ----------
        lag_max : float
            Maximum lag for prediction
        lag_min : float
            Minimum lag for prediction. Must be a negative integer.
        n_eval : int
            Number of evaluation points
        pred_vals : List[float]
            List of values for each signal to predict at. If None, uses 1.0 for all signals.
            
        Returns
        -------
        DistributedLagModelPredResults
            Container with prediction results
        """
        if lag_max is None:
            lag_max = self.nlags
        if lag_min is None:
            lag_min = self.neg_nlags
        else:
            if lag_min > 0:
                raise ValueError("lag_min must be a negative integer")
                
        if pred_vals is None:
            pred_vals = [1.0] * self.n_signals
        elif len(pred_vals) != self.n_signals:
            raise ValueError(f"pred_vals must have length {self.n_signals}")
            
        # Create evaluation points
        pred_lags = np.linspace(lag_min, lag_max, n_eval)
        
        # Create prediction basis for each signal
        pred_bases = []
        for i, basis in enumerate(self.bases):
            # Create array with n_eval rows of the prediction value
            pred_basis = dmatrix(
                basis.basis.design_info,
                {'x': pred_lags.reshape(-1, 1)}
            )
            # project prediction value on lag B-spline basis
            physio_pred = [
                pred_vals[i] * pred_basis[:, l]
                for l in range(pred_basis.shape[1])
            ]
            physio_pred = np.vstack(physio_pred).T
            pred_bases.append(physio_pred)

        # Create tensor product basis
        pred_tensor_basis = np.hstack(pred_bases)
        for i in range(len(pred_bases)):
            for j in range(i+1, len(pred_bases)):
                outer_product = self._outer_product(pred_bases[i], pred_bases[j])
                pred_tensor_basis = np.hstack([pred_tensor_basis, outer_product])

        # Get predictions
        pred_func = self.glm.predict(pred_tensor_basis)
        
        # Package results
        dlm_pred = DistributedLagModelPredResults(
            pred_func=pred_func,
            dlm_params={
                'lag_max': lag_max,
                'lag_min': lag_min,
                'n_eval': n_eval,
                'pred_lags': pred_lags,
                'pred_vals': pred_vals,
                'basis_type': self.basis_type
            }
        )
        
        return dlm_pred


def _lag_mat(x: np.ndarray, lags: list[int]) -> np.ndarray:
    """
    Create array of time-lagged copies of the time course. Modified
    for negative lags from:
    https://github.com/ulf1/lagmat
    """
    n_rows, n_cols = x.shape
    n_lags = len(lags)
    # allocate memory
    x_lag = np.empty(
        shape=(n_rows, n_cols * n_lags),
        order='F', dtype=x.dtype
    )
    # fill w/ Nans
    x_lag[:] = np.nan
    # Copy lagged columns of X into X_lag
    for i, l in enumerate(lags):
        # target columns of X_lag
        j = i * n_cols
        k = j + n_cols  # (i+1) * ncols
        # number rows of X
        nl = n_rows - abs(l)
        # Copy
        if l >= 0:
            x_lag[l:, j:k] = x[:nl, :]
        else:
            x_lag[:l, j:k] = x[-nl:, :]
    return x_lag












