import argparse
import numpy as np
import pandas as pd
import pickle

from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from utils import load_data, regress_global_signal, tr, write_cifti



def crossbasis(pvar, n_nlags, p_nlags, var_df, lag_df):
    # Construct distributed lag non-linear model cross-basis 
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2998707/
    # Create lag sequence array (include lag of 0!)
    seq_lag = np.arange(-n_nlags, p_nlags+1)
    # Create Cubic B-spline basis for predictor and lag
    basis_var = dmatrix("cr(x, df=var_df) - 1", {"x": pvar}, return_type='dataframe')
    basis_lag = dmatrix("cr(x, df=lag_df) - 1", {"x": seq_lag}, return_type='dataframe')
    # Intialize crossbasis matrix
    crossbasis = np.zeros((len(pvar), basis_var.shape[1]*basis_lag.shape[1]))
    # Loop through predictor and lag bases and multiply column pairs
    indx = 0
    for v in np.arange(basis_var.shape[1]):
        lag_mat = pd.concat([basis_var.iloc[:,v].shift(i) for i in seq_lag], axis=1)
        for l in np.arange(basis_lag.shape[1]):
            crossbasis[:, indx] = np.dot(lag_mat.values, basis_lag.iloc[:,l].values)
            indx+=1
    return crossbasis, basis_var, basis_lag


def evaluate_model(model, basis_var, basis_lag, n_nlags, p_nlags, lag_eval=20):
    # Select z-score units to evaluate physio variables
    var_pred_array = [1, 2, 3]
    # Create lag sequence array (include lag of 0!)
    seq_lag = np.arange(-n_nlags, p_nlags+1)
    # Create repeated values of pred and lag array for all possible pairwise combos
    varvec=np.repeat(var_pred_array, len(seq_lag))
    lagvec=np.tile(seq_lag, len(var_pred_array))
    # Define length of pred and lag array
    n_var = len(var_pred_array)
    n_lag = len(seq_lag)
    # Create basis from model evaluation using previously defined design matrix (for model fit)
    basis_var_pred = dmatrix(basis_var.design_info, {'x': varvec}, return_type='dataframe')
    basis_lag_pred = dmatrix(basis_lag.design_info, {'x': lagvec}, return_type='dataframe')

    # We must center our predictions around a reference value (set at 0, i.e. mean)
    cen = 0 
    # Rescale 
    basis_cen = dmatrix(basis_var.design_info, {'x': cen}, return_type='dataframe')
    basis_var_pred = basis_var_pred.subtract(basis_cen.values, axis=1)

    v_len = basis_var_pred.shape[1]
    l_len = basis_lag_pred.shape[1]
    # Row-wise kronecker product between predicted bases to generate prediction matrix
    xpred_list = [basis_var_pred.iloc[:,v]*basis_lag_pred.iloc[:,l] 
                  for v in range(v_len) for l in range(l_len)]
    xpred = pd.concat(xpred_list, axis=1)
    # Get predictions from model
    pred_mat = model.predict(xpred)
    pred_all = reshape_output(pred_mat, n_var, n_lag)
    return pred_all, var_pred_array


def ols(func, physio):
    ols_model = LinearRegression(fit_intercept=False)
    ols_model.fit(physio, func)
    return ols_model


def physio_basis(physio, n_nlags, p_nlags, lag_df=5, var_df=6):    
    # Loop through subject physio time series and convolve w/ basis
    physio_g = []
    for p in physio:
        # normalize before concatenation
        p = zscore(p)
        physio_g.append(p)
    # temporally concatenate time courses to match concatenated functional scans
    physio_g = np.concatenate(physio_g)
    # Put physio time courses into crossbasis space
    physio_p, b_var, b_lag = crossbasis(physio_g, n_nlags, p_nlags, var_df, lag_df)

    return physio_p, b_var, b_lag


def reshape_output(pred_mat, n_var, n_lag):
    # Loop through predicted map to display lags by each evaluated physio value
    pred_list = []
    indx=0
    for i in range(n_var):
        pred_mat_r = pred_mat[indx:(indx+n_lag), :]
        pred_list.append(pred_mat_r)
        indx+=n_lag
    return pred_list


def run_main(physio_label, gs, n_nlags=15, p_nlags=45):
    # n_nlags, p_nlags: lags in the negative and positive direction, respectively
    func, cifti, physio = load_data(physio_label)
    # normalize time courses
    func = zscore(func)
    # if specified, apply gs reg
    if gs:
        func = regress_global_signal(func)
    # put physio signals into spline basis
    physio_pred, basis_var, basis_lag = physio_basis(physio, n_nlags, p_nlags)
    # Lag introduces null values - trim beginning of predictor matrix
    na_indx = ~(np.isnan(physio_pred).any(axis=1))
    func = func[na_indx, :]
    physio_pred = physio_pred[na_indx,:]
    # run linear regression
    ols_res = ols(func, physio_pred)

    pred_maps, var_pred = evaluate_model(ols_res, basis_var, basis_lag, 
                                         n_nlags, p_nlags)
    out_fp = f'scan_glm_{physio_label}'
    if gs:
        out_fp += '_gs'
    cifti_meta = (cifti[0].get_fdata().shape, cifti[1], cifti[0].header)
    pickle.dump([ols_res, (n_nlags, p_nlags, pred_maps, cifti_meta)], open(f'{out_fp}.pkl', 'wb'))
    # Loop through prediction at each physio value evaluated
    for p_map, v in zip(pred_maps, var_pred):
        write_cifti(p_map, cifti, f'{out_fp}_{v}.dtseries.nii')



if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run PCA analysis')
    parser.add_argument('-p', '--physio',
                        help='<Required> physio signal',
                        choices=['resp', 'ppg'],
                        required=True,
                        type=str)
    parser.add_argument('-g', '--gs',
                        help='Whether to apply global signal regression',
                        action='store_true')


    args_dict = vars(parser.parse_args())
    run_main(args_dict['physio'], args_dict['gs'])