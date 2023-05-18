import argparse
import matplotlib.pyplot as plt
import nibabel as nb 
import nilearn.plotting as nlp
import numpy as np
import pickle
import os
import subprocess as sbp

from pathlib import Path

# Inspired by:
# https://gist.github.com/tknapen/0ebf22353130635d899eafba75de31f5


def map_to_cifti(data, cifti_meta):
    # Given 2D dataset, use a subject's cifti header to map to cifti data shape
    cifti_shape, lr_indx, cifti_hdr = cifti_meta
    cifti_data = np.zeros((data.shape[0], cifti_shape[1]), dtype=np.float32)
    brain_models = cifti_hdr.get_axis(1)  # Assume we know this
    lr_name = ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]
    for indx, s_name in zip(lr_indx, lr_name):
        for name, data_indices, model in brain_models.iter_structures():  # Iterates over volumetric and surface structures
            if name == s_name:
                cifti_data[:, data_indices] = data[:, indx]
    return cifti_data


def run_main(model_pkl, write_directory, color_map, figure_dimensions, tr, frame_rate):
    # load model object (and metadata) from glm.py function
    model, meta = pickle.load(open(model_pkl, 'rb'))
    # get figure size dimensions (x,y)
    fig_x, fig_y = split_string(figure_dimensions)
    # Get predicted maps across time points from model
    pred_maps = meta[2]
    # get maximum (abs. val) for colormap bounds
    cmax = np.max([np.abs(pred_maps.max()), np.abs(pred_maps.min())])
    cmax=0.02
    # Get time points of predicted maps
    time_vec = np.round(np.arange(-meta[0], meta[1]+1) * tr,2)

    # save out with same name as pkl file without '.pkl' extension
    input_model_base = Path(model_pkl).stem
    
    for i, data in enumerate(pred_maps):
        # Put predicted maps into cifti data matrix
        data_cifti = map_to_cifti(data[np.newaxis, :], meta[3])
        # set title as time of predicted map (in secs)
        fig_title = f'{time_vec[i]}s'
        # Create figure with two axes for left and right hemi
        fig, axes = plt.subplots(
            figsize=(fig_x, fig_y), ncols=2, subplot_kw={"projection": "3d"}
        )
        img_indx = str(i).zfill(3)
        # Get cifti brain axes from cifti header
        brain_models = meta[3][2].get_axis(1)
        _ = nlp.plot_surf("templates/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii",
                  surf_data_from_cifti(data_cifti, brain_models, 'CIFTI_STRUCTURE_CORTEX_LEFT'),
                  hemi='left', cmap=color_map, axes=axes[0], vmin=-cmax, vmax=cmax, figure=fig)
        _ = nlp.plot_surf("templates/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii",
                  surf_data_from_cifti(data_cifti, brain_models, 'CIFTI_STRUCTURE_CORTEX_RIGHT'),
                  hemi='right', cmap=color_map, axes=axes[1], vmin=-cmax, vmax=cmax, figure=fig)
        fig.suptitle(fig_title, fontsize=16)
        plt.tight_layout()
        fig.savefig(f'{write_directory}/{input_model_base}_{img_indx}.png')
        plt.close()

    # Write image to video
    write_animation_ffmpeg(input_model_base, write_directory, i, frame_rate)



def split_string(xyz_str):
    return [int(x) for x in xyz_str.split(',')]


def surf_data_from_cifti(data, axis, surf_name):
    # https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def write_animation_ffmpeg(file_base, write_directory, img_n, frame_rate):
    # Write file
    cmd = """
    ffmpeg -y -r {0} -i {2}/{1}_%03d.png -vcodec libx264 -crf 25 -acodec aac -pix_fmt yuv420p {2}/{1}.mp4
    """.format(frame_rate, file_base, write_directory)
    sbp.call(cmd, shell=True)

    # Delete PNG files in python, don't trust my self with shell :) 
    for i in range(img_n+1):
        img_indx = str(i).zfill(3)
        os.remove(f'{write_directory}/{file_base}_{img_indx}.png')




if __name__ == '__main__':
    """create nifti animation"""
    parser = argparse.ArgumentParser(description='create nifti animation')
    parser.add_argument('-n', '--input_model',
                        help='<Required> file path to model object (pkl) from glm.py function',
                        required=True,
                        type=str)
    parser.add_argument('-w', '--write_directory',
                        help='path to write static pics and animation',
                        required=True,
                        type=str)
    parser.add_argument('-c', '--color_map',
                        help='matplotlib colormap specified as string',
                        default='cold_hot',
                        type=str)
    parser.add_argument('-f', '--figure_dimensions',
                        help='x and y figure dimension size. comma separated list with no spaces',
                        default='10,6', 
                        type=str)
    parser.add_argument('-t', '--tr',
                        help='tr of functional data',
                        default=0.72, 
                        type=float)
    parser.add_argument('-r', '--frame_rate',
                        help='frame rate',
                        default=10,
                        type=float)

    args_dict = vars(parser.parse_args())
    run_main(args_dict['input_model'], args_dict['write_directory'], 
             args_dict['color_map'], args_dict['figure_dimensions'],
             args_dict['tr'], args_dict['frame_rate'])
