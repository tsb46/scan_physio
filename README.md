# scan_physio
Analysis of the relationship between the recently identified SCAN network and global BOLD fluctuations.

In a recent manuscript by Gordon et al. (2023), a set of brain regions were identified that interrupts the somatotopic organization of the primary motor cortex (the 'homunculus'). The authors refer to this set of regions as the somato-cognitive action network (SCAN).  

In an open review by the Diedrichsen lab  (https://www.diedrichsenlab.org/BrainDataScience/or_gordon2023/index.htm), it is suggested that the co-activation of these brain regions may reflect muscle synergies induced by respiratory motion - i.e. abdmonimal, larynx and upper face muscular activity. To further assess these claims we analyzed the relationship between these regions and the amplitude of respiratory belt signals. This analysis was conduced on a small subset of randomly sampled HCP subjects (N=20) with respiratory belt recordings of sufficient quality. This code contains a set of command-line python scripts for replicating the analysis. The code was run with Python 3.11.3.

# Installation
To run the code in this repository, several Python packages must be installed in your virtual environment with the following command:
```
pip install -r requirements.txt
```

# Pulling HCP data
To pull HCP resting-state recordings (and physio), we use the boto3 package. Simply run the following in your terminal to download the data:
```
python pull_data.py
```

# Preprocessing Data
In the pull_data.py script, we pulled MSMAll-registered cifti files for the HCP resting-state recordings that have been previously preprocessed with ICA-FIX by the HCP team. Additional preprocessing is implemented for surface smoothing (4mm FWHM), z-score normalization (of vertex time courses) and band-pass filtering into the conventional resting-state BOLD frequency range (0.01-0.1Hz). To run the preprocessing script, run the following command in your terminal:
```
python preprocess.py
```

# Command-Line scripts
To replicate our analyses, a set of command-line Python scripts are provided. The command-line interface comes with help documentation for each parameter. To replicate our analyses, run the following commands:

* For principal component analysis (the first principal component corresponds to the 'global signal'):
```
python pca.py -n 10
```
* For traditional whole-brain GLM analysis of respiratory belt amplitudes regressed on all vertex time courses, use the following command:

```
python glm.py -p resp
```
Note, cubic spline bases of lagged respiratory belt time courses are used as regressors. Also note, no statistical testing is performed on these maps. The output of this analysis is a pickled (scikit-learn) model object (and meta parameters), along with predicted BOLD time courses at regularly spaced lags of the respiratory belt amplitude signal (the amplitude of the respiratory belt amplitude time course for prediction is set at one standard deviation above the mean - i.e. z-score-1). 

* For non-linear whole-brain GLM analyses of the respiratory belt amplitudes, using a distributed lag non-linear model (DLNM; Gasparrini et al. 2013), run the following command:

```
python glm_dlnm.py -p resp
```
Note, cubic spline basis for both the lagged respiratory belt time course and its amplitude values is used as regressors. Also note, no statistical testing is performed on these maps. The output of this analysis is a pickled (scikit-learn) model object (and meta parameters), along with predicted BOLD time courses at regularly spaced lags of the respiratory belt amplitude signal. To assess the response at different amplitudes of the respiratory belt amplitude signal, we assess the predicted BOLD time courses at several amplitude values - z-score = 1, 2, and 3. 

# Citations
Gasparrini, A., Armstrong, B., & Kenward, M. G. (2010). Distributed lag non-linear models. Statistics in Medicine, 29(21), 2224–2234. https://doi.org/10.1002/sim.3940


Gordon, E. M., Chauvin, R. J., Van, A. N., Rajesh, A., Nielsen, A., Newbold, D. J., Lynch, C. J., Seider, N. A., Krimmel, S. R., Scheidter, K. M., Monk, J., Miller, R. L., Metoki, A., Montez, D. F., Zheng, A., Elbau, I., Madison, T., Nishino, T., Myers, M. J., … Dosenbach, N. U. F. (2023). A somato-cognitive action network alternates with effector regions in motor cortex. Nature, 617(7960), Article 7960. https://doi.org/10.1038/s41586-023-05964-2






