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

5) Complex PCA analysis of functional MRI data.

6) Functional connectivity analysis of ROI time series.


Note, for the Vanderbilt dataset, the respiratory belt signal had lost signal at certain
points in time. These time points are identified and removed from the analysis using a
weighting vector.
"""

import argparse
import os
import pickle
import random
from collections import defaultdict

from typing import Tuple, List

import numpy as np

from scan.io.load import DatasetLoad, Gifti
from scan.model.fc import FCMapModel, FCInteractionModel
from scan.model.glm import DistributedLagModel, MultivariateDistributedLagModel
from scan.model.complex_pca import ComplexPCA

# OUTPUT DIRECTORY
OUT_DIRECTORY = "results/main"

# SCAN ROI MASKS (fs_LR)
SCAN_LH_ROI_MASKS = [
    "template/lh.SCAN_manual_bottom.label.gii",
    "template/lh.SCAN_manual_middle.label.gii",
    "template/lh.SCAN_manual_top.label.gii",
]
SCAN_RH_ROI_MASKS = [
    "template/rh.SCAN_manual_bottom.label.gii",
    "template/rh.SCAN_manual_middle.label.gii",
    "template/rh.SCAN_manual_top.label.gii",
]
# SCAN ATLAS ROI
SCAN_LH_ATLAS_ROI_MASK = "template/lh.SCAN_fsLR.label.gii"
SCAN_RH_ATLAS_ROI_MASK = "template/rh.SCAN_fsLR.label.gii"

# Effector Atlas ROIs (fs_LR)
FOOT_LH_ROI_MASK = "template/lh.FOOT_fsLR.label.gii"
FOOT_RH_ROI_MASK = "template/rh.FOOT_fsLR.label.gii"
HAND_LH_ROI_MASK = "template/lh.HAND_fsLR.label.gii"
HAND_RH_ROI_MASK = "template/rh.HAND_fsLR.label.gii"
MOUTH_LH_ROI_MASK = "template/lh.MOUTH_fsLR.label.gii"
MOUTH_RH_ROI_MASK = "template/rh.MOUTH_fsLR.label.gii"


def load_data(
    analysis: str,
    roi: bool = False,
    left_roi_masks: List[str] | None = None,
    right_roi_masks: List[str] | None = None,
) -> Tuple[DatasetLoad, dict, Gifti]:
    """
    Load data from dataset.
    """
    if roi and (left_roi_masks is None or right_roi_masks is None):
        raise ValueError("Must provide left and right ROI masks if roi is True")

    # determine whether to concat functional data across sessions based on
    # whether we're doing whole-brain or ROI-based analysis
    if analysis in [
        "whole-brain-univariate",
        "whole-brain-multivariate",
        "complex-pca",
        "fc",
    ]:
        concat = True
    else:
        concat = False

    # load all scans in dataset
    loader = DatasetLoad("vanderbilt")
    # load data with ROI masks if specified
    if roi:
        data, gii = loader.load(
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            # high-pass filter physiological data w/ 0.01 Hz cutoff
            physio_high_pass=True,
            input_mask=True,
            # average left and right hemisphere ROI time courses together
            roi_avg_left_right=True,
            lh_roi_masks=left_roi_masks,
            rh_roi_masks=right_roi_masks,
            concat=concat,
        )
        # get average of emg and eog channels
        eog_mean, emg_mean = _physio_average(data, concat)
    else:
        data, gii = loader.load(
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            # high-pass filter physiological data w/ 0.01 Hz cutoff
            physio_high_pass=True,
            concat=concat,
        )
        # get average of emg and eog channels
        eog_mean, emg_mean = _physio_average(data, concat)

    data_out = {
        "func": data["func"],
        "eog": eog_mean,
        "emg": emg_mean,
        "resp": data["physio"]["resp_amp"],
        "weight": data["physio"]["weight"],
    }
    return loader, data_out, gii


def main(analysis: str, out_dir: str, physio: str):
    # create out directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    if analysis == "whole-brain-univariate":
        loader, data, gii = load_data(analysis=analysis, roi=False)
        whole_brain_univariate(data, gii, physio, out_dir)
    elif analysis == "whole-brain-multivariate":
        loader, data, gii = load_data(analysis=analysis, roi=False)
        whole_brain_multivariate(data, gii, out_dir)
    elif analysis == "roi-univariate":
        loader, data, gii = load_data(
            analysis=analysis,
            roi=True,
            left_roi_masks=SCAN_LH_ROI_MASKS
            + [FOOT_LH_ROI_MASK, HAND_LH_ROI_MASK, MOUTH_LH_ROI_MASK],
            right_roi_masks=SCAN_RH_ROI_MASKS
            + [FOOT_RH_ROI_MASK, HAND_RH_ROI_MASK, MOUTH_RH_ROI_MASK],
        )
        roi_univariate(data, gii, loader, physio, out_dir)
    elif analysis == "roi-multivariate":
        loader, data, gii = load_data(
            analysis=analysis,
            roi=True,
            left_roi_masks=SCAN_LH_ROI_MASKS
            + [FOOT_LH_ROI_MASK, HAND_LH_ROI_MASK, MOUTH_LH_ROI_MASK],
            right_roi_masks=SCAN_RH_ROI_MASKS
            + [FOOT_RH_ROI_MASK, HAND_RH_ROI_MASK, MOUTH_RH_ROI_MASK],
        )
        roi_multivariate(data, gii, loader, physio, out_dir)
    elif analysis == "complex-pca":
        loader, data, gii = load_data(analysis=analysis, roi=False)
        complex_pca(data, gii, out_dir)
    elif analysis == "fc":
        # for seed-based FC analysis, we need to load ROI time courses and full cortical time series
        # load SCAN atlas ROI time courses
        _, data_roi_scan, _ = load_data(
            analysis=analysis,
            roi=True,
            left_roi_masks=[SCAN_LH_ATLAS_ROI_MASK],
            right_roi_masks=[SCAN_RH_ATLAS_ROI_MASK],
        )
        # average left and right hemisphere atlas ROI time courses to get single time course for each ROI
        seed_scan_roi_ts = data_roi_scan["func"].mean(axis=1)[:, np.newaxis]

        # load full cortical time series
        loader, data, gii = load_data(analysis=analysis, roi=False)

        # perform seed-based FC analysis using SCAN atlas ROIs as seeds
        roi_fc(seed_scan_roi_ts, data, gii, loader, "scan", out_dir)


def complex_pca(data: dict, gii: Gifti, out_dir: str):
    # perform complex PCA on the functional data
    cpca = ComplexPCA(n_components=5)
    cpca_res = cpca.decompose(data["func"])
    cpca_res.write(gii, file_prefix="vanderbilt_cpca", out_dir=out_dir)
    for i in range(3):
        cpca_recon = cpca.reconstruct(i)
        cpca_recon.write(
            gii, file_prefix=f"vanderbilt_cpca_recon_n{i + 1}", out_dir=out_dir
        )


def roi_fc(
    seed_roi_ts: np.ndarray,
    data: dict,
    gii: Gifti,
    loader: DatasetLoad,
    label: str,
    out_dir: str,
):
    global_signal_ts = data["func"].mean(axis=1)[:, np.newaxis]

    # fit FC map model to predict time series of each ROI from time series of all other ROIs
    fc_map_model = FCMapModel()
    fc_map_res = fc_map_model.fit(seed_ts=seed_roi_ts, func_data=data["func"])
    fc_map_res.write(gii, file_prefix=f"vanderbilt_fc_map_{label}_roi", out_dir=out_dir)

    # fit FC interaction model to predict time series of each ROI from time series of all other ROIs, modulated by physiological signal
    fc_interaction_model = FCInteractionModel()
    fc_interaction_res = fc_interaction_model.fit(
        seed_ts=seed_roi_ts,
        func_data=data["func"],
        modulator=global_signal_ts,
        weights=np.squeeze(data["weight"].reshape(-1, 1)),
    )
    fc_interaction_res.write(
        gii,
        file_prefix=f"vanderbilt_fc_interaction_{label}_roi_global_signal",
        out_dir=out_dir,
    )


def roi_univariate(
    data: dict,
    gii: Gifti,
    loader: DatasetLoad,
    physio: str,
    out_dir: str,
):
    # generate bootstrap samples
    subject_session_list = list(loader.iter)
    bootstrap_samples = _generate_bootstrap_samples(subject_session_list)
    bootstrap_preds = []
    # fit model to each bootstrap sample
    for i, sample in enumerate(bootstrap_samples):
        print(f"Bootstrap sample {i + 1} of {len(bootstrap_samples)}")
        data_out = _concat_bootstrap_samples(sample, subject_session_list, data)
        dlm = DistributedLagModel(nlags=10, neg_nlags=-5, n_knots=5, basis="cr")
        dlm.fit(
            data_out[physio],
            data_out["func"],
            weights=np.squeeze(data_out["weight"].reshape(-1, 1)),
        )
        pred_func = dlm.evaluate()
        bootstrap_preds.append(pred_func.pred_func)
    # compute mean and std of bootstrap predictions
    bootstrap_preds_mean = np.mean(bootstrap_preds, axis=0)
    bootstrap_preds_std = np.std(bootstrap_preds, axis=0)
    # save results
    roi_pred_out = {
        "pred": bootstrap_preds_mean,
        "std": bootstrap_preds_std,
        "physio_labels": [physio],
        "roi": loader.roi_names,
        "pred_lags": pred_func.dlm_params["pred_lags"],  # type: ignore
    }
    with open(os.path.join(out_dir, f"{physio}_roi_dlm.pkl"), "wb") as f:
        pickle.dump(roi_pred_out, f)


def roi_multivariate(
    data: dict,
    gii: Gifti,
    loader: DatasetLoad,
    physio: str,
    out_dir: str,
):
    subject_session_list = list(loader.iter)
    bootstrap_samples = _generate_bootstrap_samples(subject_session_list)
    bootstrap_preds = {
        "v000": [],
        "v200": [],
        "v020": [],
        "v002": [],
        "v220": [],
        "v202": [],
        "v022": [],
        "v222": [],
    }
    for i, sample in enumerate(bootstrap_samples):
        print(f"Bootstrap sample {i + 1} of {len(bootstrap_samples)}")
        data_out = _concat_bootstrap_samples(sample, subject_session_list, data)
        mdlm = MultivariateDistributedLagModel(
            nlags=10,
            neg_nlags=-5,
            n_knots=5,
            basis="cr",
        )
        mdlm.fit(
            np.hstack([data_out["resp"], data_out["eog"], data_out["emg"]]),
            data_out["func"],
            weights=np.squeeze(data_out["weight"].reshape(-1, 1)),
        )
        # evaluate model at different combinations of physiological signal values
        pred_func = mdlm.evaluate(pred_vals=[0, 0, 0])
        bootstrap_preds["v000"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[2, 0, 0])
        bootstrap_preds["v200"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[0, 2, 0])
        bootstrap_preds["v020"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[0, 0, 2])
        bootstrap_preds["v002"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[2, 2, 0])
        bootstrap_preds["v220"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[2, 0, 2])
        bootstrap_preds["v202"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[0, 2, 2])
        bootstrap_preds["v022"].append(pred_func.pred_func)
        pred_func = mdlm.evaluate(pred_vals=[2, 2, 2])
        bootstrap_preds["v222"].append(pred_func.pred_func)

    # compute mean and std of bootstrap predictions
    bootstrap_preds_mean = {
        "v200": np.mean(bootstrap_preds["v200"], axis=0),
        "v020": np.mean(bootstrap_preds["v020"], axis=0),
        "v002": np.mean(bootstrap_preds["v002"], axis=0),
        "v220": np.mean(bootstrap_preds["v220"], axis=0),
        "v202": np.mean(bootstrap_preds["v202"], axis=0),
        "v022": np.mean(bootstrap_preds["v022"], axis=0),
        "v222": np.mean(bootstrap_preds["v222"], axis=0),
    }
    bootstrap_preds_std = {
        "v200": np.std(bootstrap_preds["v200"], axis=0),
        "v020": np.std(bootstrap_preds["v020"], axis=0),
        "v002": np.std(bootstrap_preds["v002"], axis=0),
        "v220": np.std(bootstrap_preds["v220"], axis=0),
        "v202": np.std(bootstrap_preds["v202"], axis=0),
        "v022": np.std(bootstrap_preds["v022"], axis=0),
        "v222": np.std(bootstrap_preds["v222"], axis=0),
    }

    with open(os.path.join(out_dir, "roi_mdlm.pkl"), "wb") as f:
        pickle.dump(
            {
                "pred": bootstrap_preds_mean,
                "std": bootstrap_preds_std,
                "physio_labels": ["resp", "eog", "emg"],
                "roi": loader.roi_names,
                "pred_lags": pred_func.dlm_params["pred_lags"],  # type: ignore
            },
            f,
        )


def whole_brain_univariate(data: dict, gii: Gifti, physio: str, out_dir: str):
    dlm = DistributedLagModel(nlags=10, neg_nlags=-5, n_knots=5, basis="cr")
    dlm.fit(
        data[physio], data["func"], weights=np.squeeze(data["weight"].reshape(-1, 1))
    )
    pred_func = dlm.evaluate()
    pred_func.write(gii, file_prefix=f"vanderbilt_dlm_{physio}", out_dir=out_dir)


def whole_brain_multivariate(data: dict, gii: Gifti, out_dir: str):
    mdlm = MultivariateDistributedLagModel(
        nlags=10,
        neg_nlags=-5,
        n_knots=5,
        basis="cr",
    )

    mdlm.fit(
        np.hstack([data["resp"], data["eog"], data["emg"]]),
        data["func"],
        weights=np.squeeze(data["weight"].reshape(-1, 1)),
    )
    # evaluate model at different combinations of physiological signal values
    pred_func = mdlm.evaluate(pred_vals=[0, 0, 0])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v000", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[2, 0, 0])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v200", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[0, 2, 0])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v020", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[0, 0, 2])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v002", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[2, 2, 0])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v220", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[2, 0, 2])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v202", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[0, 2, 2])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v022", out_dir=out_dir
    )

    pred_func = mdlm.evaluate(pred_vals=[2, 2, 2])
    pred_func.write(
        gii, file_prefix="vanderbilt_mdlm_resp_eog_emg_v222", out_dir=out_dir
    )


def _physio_average(data: dict, concat: bool) -> Tuple[list[float], list[float]]:
    """
    Average emg and eog physiological signals across channels.
    """
    if not concat:
        eog_mean = [
            (eog1 + eog2) / 2
            for eog1, eog2 in zip(
                data["physio"]["eog1_amp"], data["physio"]["eog2_amp"]
            )
        ]
        emg_mean = [
            (emg1 + emg2 + emg3) / 3
            for emg1, emg2, emg3 in zip(
                data["physio"]["emg1_amp"],
                data["physio"]["emg2_amp"],
                data["physio"]["emg3_amp"],
            )
        ]
    else:
        eog = [data["physio"]["eog1_amp"], data["physio"]["eog2_amp"]]
        eog_mean = np.mean(np.hstack(eog), axis=1)[:, np.newaxis].tolist()
        emg = [
            data["physio"]["emg1_amp"],
            data["physio"]["emg2_amp"],
            data["physio"]["emg3_amp"],
        ]
        emg_mean = np.mean(np.hstack(emg), axis=1)[:, np.newaxis].tolist()

    return eog_mean, emg_mean


def _concat_bootstrap_samples(
    bootstrap_sample: List[Tuple[str, str]],
    subject_session_list: List[Tuple[str, str]],
    data: dict,
) -> dict:
    """
    Concatenate bootstrap samples into a single dataset.
    """
    # get indices of bootstrap samples
    bootstrap_indices = [
        subject_session_list.index((subject, session))
        for subject, session in bootstrap_sample
    ]
    # concatenate bootstrap samples
    data_out = {}
    for key in data.keys():
        data_out[key] = np.concatenate(
            [data[key][index] for index in bootstrap_indices], axis=0
        )
    return data_out


def _generate_bootstrap_samples(
    subject_session_list: List[Tuple[str, str]], n_bootstrap: int = 1000
) -> List[List[Tuple[str, str]]]:
    """
    Generate bootstrap samples at the subject level.

    This function takes a list of (subject, session) tuples and generates
    repeated samples with replacement of subjects. Multiple sessions per subject
    are preserved in the bootstrap samples.

    Parameters
    ----------
    subject_session_list : List[Tuple[str, str]]
        List of (subject, session) tuples
    n_bootstrap : int, optional
        Number of bootstrap samples to generate (default: 1000)

    Returns
    -------
    List[List[Tuple[str, str]]]
        List of bootstrap samples, where each sample is a list of
        (subject, session) tuples
    """
    # Group sessions by subject
    subject_sessions = defaultdict(list)
    for subject, session in subject_session_list:
        subject_sessions[subject].append(session)

    # Get unique subjects
    unique_subjects = list(subject_sessions.keys())
    n_subjects = len(unique_subjects)

    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Sample subjects with replacement
        sampled_subjects = random.choices(unique_subjects, k=n_subjects)

        # Create bootstrap sample by including all sessions for each sampled subject
        bootstrap_sample = []
        for subject in sampled_subjects:
            for session in subject_sessions[subject]:
                bootstrap_sample.append((subject, session))

        bootstrap_samples.append(bootstrap_sample)

    return bootstrap_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replication of SCAN network analysis")
    parser.add_argument(
        "-a",
        "--analysis",
        type=str,
        required=True,
        help="Analysis to perform",
        choices=[
            "whole-brain-univariate",
            "whole-brain-multivariate",
            "roi-univariate",
            "roi-multivariate",
            "complex-pca",
            "fc",
        ],
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=False,
        help="output directory to save analysis outputs",
        default=OUT_DIRECTORY,
    )
    parser.add_argument(
        "-p",
        "--physio",
        type=str,
        required=False,
        help="Physiological signal to use for univariate analysis and physio-modulated FC analysis",
        choices=["eog", "emg", "resp"],
        default="resp",
    )
    args = parser.parse_args()
    main(args.analysis, args.out_dir, args.physio)
