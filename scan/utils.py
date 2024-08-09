"""
Utility module
"""

import os


def get_fp_base(fp: str) -> str:
    """
    get nifti file path without extenstion
    """
    fp_split = os.path.splitext(fp)
    if fp_split[1] == '.gz':
        fp_base = os.path.splitext(fp_split[0])[0]
    else:
        fp_base = fp_split[0]
    return fp_base



