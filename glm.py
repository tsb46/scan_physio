import argparse
import numpy as np
import pandas as pd
import pickle

from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from utils import load_data, regress_global_signal, tr, write_cifti


def evaluate_model(basis, model, n_nlag, p_nlag,
                   p_eval=1, n_eval=20):
    # Set lag vector for physio basis
    lag_vec = np.arange(-n_nlag, p_nlag+1)
    pred_basis = dmatrix(basis.design_info, {'x': lag_vec},
                                return_type='dataframe')
     # Intialize basis matrix
    pred_list = [p_eval * pred_basis.iloc[:, l].values 
                 for l in range(pred_basis.shape[1])]
    physio_pred = np.vstack(pred_list).T
    # Get predictions from model
    pred_bold = model.predict(physio_pred)
    return pred_bold


def ols(func, physio):
    ols_model = LinearRegression(fit_intercept=False)
    ols_model.fit(physio, func)
    return ols_model


def physio_basis(physio, n_nlag, p_nlag, lag_df=5):
    # generate physio basis with splines along lag 
    # Create vector of lags
    lag_vec = np.arange(-n_nlag, p_nlag+1)
    # create cubic spline basis
    basis = dmatrix("cr(x, df=lag_df) - 1", {"x": lag_vec}, 
                    return_type='dataframe')
    # Loop through subject physio time series and convolve w/ basis
    physio_hrf = []
    for p in physio:
        p = zscore(p)
        p = pd.Series(np.squeeze(p))
        # Lag event_ts by lags in lag_vec
        lag_mat = pd.concat([p.shift(l, fill_value=0) for l in lag_vec], 
                        axis=1).values
        # Intialize regressor matrix
        regressor_mat = np.zeros((len(p), basis.shape[1]))
        # Loop through splines bases and multiply with lagged event time course
        for l in np.arange(basis.shape[1]):
            regressor_mat[:, l] = np.dot(lag_mat, basis.iloc[:,l].values)
        # normalize before concatenation
        physio_hrf.append(regressor_mat)

    # temporally concatenate convolved time courses to match concatenated functional scans
    physio_hrf = np.concatenate(physio_hrf)
    return physio_hrf, basis


def run_main(physio_label, gs, n_nlag=15, p_nlag=45):
    func, cifti, physio = load_data(physio_label)
    # normalize time courses
    func = zscore(func)
    # if specified, apply gs reg
    if gs:
        func = regress_global_signal(func)
    # convolve physio signals with hrf basis
    physio_hrf, basis = physio_basis(physio, n_nlag, p_nlag)
    # run linear regression
    ols_res = ols(func, physio_hrf)
    pred_maps = evaluate_model(basis, ols_res, n_nlag, p_nlag)
    out_fp = f'scan_glm_{physio_label}'
    if gs:
        out_fp += '_gs'
    # get cifti parameters for later analytics
    cifti_meta = (cifti[0].get_fdata().shape, cifti[1], cifti[0].header)
    pickle.dump([ols_res, (n_nlag, p_nlag, pred_maps, cifti_meta)], open(f'{out_fp}.pkl', 'wb'))
    write_cifti(pred_maps, cifti, f'{out_fp}.dtseries.nii')





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