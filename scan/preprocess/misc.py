from typing import List

from tedana.workflows import tedana_workflow



def tedana_denoise(
    fps_in: str,
    echo_times: List[float],
    mask: str,
    out_dir: str,
    out_prefix: str,
):
    """
    Run tedana workflow - Multi-Echo ICA denoising

    Parameters
    ----------
        fps_in: str
            lists of filepaths to each echo in order
        echo_times: List[float]
            echo times in order
        mask: str
            file path to freesurfer binary brain mask in functional space
        out_dir: str
            output directory
        out_prefix: str
            file name prefix for all tedana outputs
    """
    # run tedana workflow
    tedana_workflow(
        data=fps_in, tes=echo_times, mask=mask,
        prefix=out_prefix, out_dir=out_dir,
        overwrite=True
    )
