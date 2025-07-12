"""
Command-line utility for performing replication of SCAN network 
analysis. Several analysis pipelines are available, including:

1) Whole-brain univariate analysis of physiological signals (respiration,
electromyography, and electrooculography) regressed onto cortical time
series. 

2) Whole-brain multivariate analysis of physiological signals (respiration,
electromyography, and electrooculography) regressed onto cortical time series. 

3) ROI-based univariate analysis of physiological signals regressed onto
cortical time series. 

4) ROI-based multivariate analysis of physiological signals regressed onto
cortical time series. 

Note, for the Vanderbilt dataset, the respiratory belt signal had lost signal at certain
points in time. These time points are identified and removed from the analysis using a 
weighting vector.
"""

import argparse
import os
import pickle

from typing import Literal, Tuple

import numpy as np

from scan.io.load import DatasetLoad, Gifti
from scan.model.corr import (
    DistributedLagModel, 
    MultivariateDistributedLagModel
)

# OUTPUT DIRECTORY
OUT_DIRECTORY = 'results/main'

# SCAN ROI MASKS (fs_LR)
LH_ROI_MASKS = [
    'template/scan_roi_lh_BOTTOM.label.gii',
    'template/scan_roi_lh_MIDDLE.label.gii',
    'template/scan_roi_lh_TOP.label.gii'
]
RH_ROI_MASKS = [
    'template/scan_roi_rh_BOTTOM.label.gii',
    'template/scan_roi_rh_MIDDLE.label.gii',
    'template/scan_roi_rh_TOP.label.gii'
]
ROI_MASK_LABELS = {
    'lh': {
        'BOTTOM': 'template/scan_roi_lh_BOTTOM.label.gii',
        'MIDDLE': 'template/scan_roi_lh_MIDDLE.label.gii',
        'TOP': 'template/scan_roi_lh_TOP.label.gii'
    },
    'rh': {
        'BOTTOM': 'template/scan_roi_rh_BOTTOM.label.gii',
        'MIDDLE': 'template/scan_roi_rh_MIDDLE.label.gii',
        'TOP': 'template/scan_roi_rh_TOP.label.gii'
    }
}

def main(analysis: str, out_dir: str, physio: str):
    # create out directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    if analysis == 'whole-brain-univariate':
        loader, data, gii = load_data(roi=False)
        whole_brain_univariate(data, gii, physio, out_dir)
    elif analysis == 'whole-brain-multivariate':
        loader, data, gii = load_data(roi=False)
        whole_brain_multivariate(data, gii, out_dir)
    elif analysis == 'roi-univariate':
        loader, data, gii = load_data(roi=True)
        roi_univariate(data, gii, loader, physio, out_dir)
    elif analysis == 'roi-multivariate':
        loader, data, gii = load_data(roi=True)
        roi_multivariate(data, gii, loader, physio, out_dir)


def roi_univariate(
    data: dict, 
    gii: Gifti, 
    loader: DatasetLoad,
    physio: str, 
    out_dir: str,
):
    dlm = DistributedLagModel(nlags=10, neg_nlags=-5, n_knots=5, basis='cr')
    dlm.fit(
        data[physio],
        data['func'], 
        weights=np.squeeze(data['weight'].reshape(-1,1))
    )
    pred_func = dlm.evaluate()
    roi_pred_out = {
        'pred': pred_func.pred_func,
        'roi': loader.roi_names,
        'params': pred_func.dlm_params
    }
    with open(os.path.join(out_dir, f'{physio}_roi_dlm.pkl'), 'wb') as f:
        pickle.dump(roi_pred_out, f)


def roi_multivariate(
    data: dict, 
    gii: Gifti, 
    loader: DatasetLoad,
    physio: str, 
    out_dir: str,
):
    mdlm = MultivariateDistributedLagModel(
        nlags=10, neg_nlags=-5, n_knots=5, basis='cr',
    )

    mdlm.fit(
        np.hstack([data['resp'], data['eog'], data['emg']]),
        data['func'],
        weights=np.squeeze(data['weight'].reshape(-1,1))
    )
    # evaluate model at different combinations of physiological signal values
    pred_func_000 = mdlm.evaluate(pred_vals=[0,0,0])
    pred_func_200 = mdlm.evaluate(pred_vals=[2,0,0])
    pred_func_020 = mdlm.evaluate(pred_vals=[0,2,0])
    pred_func_002 = mdlm.evaluate(pred_vals=[0,0,2])
    pred_func_220 = mdlm.evaluate(pred_vals=[2,2,0])
    pred_func_202 = mdlm.evaluate(pred_vals=[2,0,2])
    pred_func_022 = mdlm.evaluate(pred_vals=[0,2,2])
    pred_func_222 = mdlm.evaluate(pred_vals=[2,2,2])

    roi_pred_out = {
        'pred': {
            'v000': pred_func_000.pred_func,
            'v200': pred_func_200.pred_func,
            'v020': pred_func_020.pred_func,
            'v002': pred_func_002.pred_func,
            'v220': pred_func_220.pred_func,
            'v202': pred_func_202.pred_func,
            'v022': pred_func_022.pred_func,
            'v222': pred_func_222.pred_func
        },
        'physio_labels': ['resp', 'eog', 'emg'],
        'roi': loader.roi_names,
        'params': {
            'v000': pred_func_000.dlm_params,
            'v200': pred_func_200.dlm_params,
            'v020': pred_func_020.dlm_params,
            'v002': pred_func_002.dlm_params,
            'v220': pred_func_220.dlm_params,
            'v202': pred_func_202.dlm_params,
            'v022': pred_func_022.dlm_params,
            'v222': pred_func_222.dlm_params
        }
    }
    with open(os.path.join(out_dir, f'roi_mdlm.pkl'), 'wb') as f:
        pickle.dump(roi_pred_out, f)


def whole_brain_univariate(
    data: dict, 
    gii: Gifti, 
    physio: str, 
    out_dir: str
):
    dlm = DistributedLagModel(nlags=10, neg_nlags=-5, n_knots=5, basis='cr')
    dlm.fit(
        data[physio],
        data['func'], 
        weights=np.squeeze(data['weight'].reshape(-1,1))
    )
    pred_func = dlm.evaluate()
    pred_func.write(
        gii, 
        file_prefix=f'vanderbilt_dlm_{physio}', 
        out_dir=out_dir
    )


def whole_brain_multivariate(
    data: dict, 
    gii: Gifti, 
    out_dir: str
):
    mdlm = MultivariateDistributedLagModel(
        nlags=10, neg_nlags=-5, n_knots=5, basis='cr',
    )

    mdlm.fit(
        np.hstack([data['resp'], data['eog'], data['emg']]),
        data['func'],
        weights=np.squeeze(data['weight'].reshape(-1,1))
    )
    # evaluate model at different combinations of physiological signal values
    pred_func = mdlm.evaluate(pred_vals=[0,0,0])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v000', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[2,0,0])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v200', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[0,2,0])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v020', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[0,0,2])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v002', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[2,2,0])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v220', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[2,0,2])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v202', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[0,2,2])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v022', out_dir=out_dir)

    pred_func = mdlm.evaluate(pred_vals=[2,2,2])
    pred_func.write(gii, file_prefix='vanderbilt_mdlm_resp_eog_emg_v222', out_dir=out_dir)


def load_data(roi: bool = False) -> Tuple[DatasetLoad, dict, Gifti]:
    """
    Load data from dataset.
    """
    # load all scans in dataset
    loader = DatasetLoad(
        'vanderbilt'
    )
    # load data with ROI masks if specified
    if roi:
        data, gii = loader.load(
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            # high-pass filter physiological data w/ 0.01 Hz cutoff
            physio_high_pass=True,
            input_mask=True,
            lh_roi_masks=LH_ROI_MASKS,
            rh_roi_masks=RH_ROI_MASKS
        )
    else:
        data, gii = loader.load(
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            # high-pass filter physiological data w/ 0.01 Hz cutoff
            physio_high_pass=True,
        )
    
    # get average of emg and eog channels
    eog = [
        data['physio']['eog1_amp'],
        data['physio']['eog2_amp']
    ]
    eog_mean = np.mean(np.hstack(eog), axis=1)[:,np.newaxis]
    emg = [
        data['physio']['emg1_amp'],
        data['physio']['emg2_amp'],
        data['physio']['emg3_amp']
    ]
    emg_mean = np.mean(np.hstack(emg), axis=1)[:,np.newaxis]
    data_out = {
        'func': data['func'],
        'eog': eog_mean,
        'emg': emg_mean,
        'resp': data['physio']['resp_amp'],
        'weight': data['physio']['weight']
    }
    return loader, data_out, gii


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replication of SCAN network analysis'
    )
    parser.add_argument(
        '-a',
        '--analysis',
        type=str,
        required=True,
        help='Analysis to perform',
        choices=[
            'whole-brain-univariate',
            'whole-brain-multivariate',
            'roi-univariate',
            'roi-multivariate'
        ]
    )
    parser.add_argument(
        '-o',
        '--out_dir',
        type=str,
        required=False,
        help="output directory to save analysis outputs",
        default=OUT_DIRECTORY
    )
    parser.add_argument(
        '-p',
        '--physio',
        type=str,
        required=False,
        help='Physiological signal to use for univariate analysis',
        choices=['eog', 'emg', 'resp'],
        default='resp'
    )
    args = parser.parse_args()
    main(args.analysis, args.out_dir, args.physio)