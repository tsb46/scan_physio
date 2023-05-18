import argparse
import numpy as np
import pickle

from factor_analyzer import Rotator
from numpy.linalg import pinv
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import zscore
from utils import load_data, regress_global_signal, write_cifti


def rotate_loadings(func, loadings, rotation='promax'):
    # initialize FactorAnalyzer rotation object
    rotator = Rotator(rotation)
    # rotate loadings
    loadings_r = rotator.fit_transform(loadings)
    # compute new pc scores from rotated loadings
    # https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
    pc_scores_r = func @ pinv(loadings_r).T
    return loadings_r, pc_scores_r


def run_main(n_comps, rotate, gs):
    func, cifti, _ = load_data(None)
    # normalize time courses
    func = zscore(func)
    # if specified, apply gs reg
    if gs:
        func = regress_global_signal(func)
    # initialize PCA
    pca = PCA(n_components=n_comps)
    # fit and get PC scores
    pc_scores = pca.fit_transform(func)
    # convert loadings to 'correlations'
    # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    loadings =  pca.components_.T @ np.diag(pca.singular_values_) 
    loadings /= np.sqrt(func.shape[0]-1)
    # apply rotation, if specified
    if rotate:
        loadings, pc_scores = rotate_loadings(func, loadings)
    # package pca results
    pca_dict = {
        'exp_var': pca.explained_variance_ratio_,
        'scores': pc_scores,
        'comps': pca.components_,
        'loadings': loadings
    }
    out_fp = 'scan_pc'
    if rotate:
        out_fp += '_promax'
    if gs:
        out_fp += '_gs'
    pickle.dump(pca_dict, open(f'{out_fp}.pkl', 'wb'))
    write_cifti(loadings.T, cifti, f'{out_fp}.dtseries.nii')



if __name__ == '__main__':
    """Run main analysis"""
    parser = argparse.ArgumentParser(description='Run PCA analysis')
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=True,
                        type=int)
    parser.add_argument('-r', '--rotate',
                        help='Whether to apply promax rotation',
                        action='store_true')
    parser.add_argument('-g', '--gs',
                        help='Whether to apply global signal regression',
                        action='store_true')


    args_dict = vars(parser.parse_args())
    run_main(args_dict['n_comps'], args_dict['rotate'], args_dict['gs'])