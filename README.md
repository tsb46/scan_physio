# scan_physio
Analysis of the relationship between the recently identified SCAN network and global BOLD fluctuations.

In a recent manuscript by Gordon et al. (2023), a set of brain regions were identified that interrupts the somatotopic organization of the primary motor cortex (the 'homunculus'). The authors refer to this set of regions as the somato-cognitive action network (SCAN).  

In an open review by the Diedrichsen lab  (https://www.diedrichsenlab.org/BrainDataScience/or_gordon2023/index.htm), it is suggested that the co-activation of these brain regions may reflect muscle synergies induced by respiratory motion - i.e. abdmonimal, larynx and upper face muscular activity. To further assess these claims we analyzed the relationship between these regions and the amplitude of respiratory belt signals. This analysis was conduced on a small subset of randomly sampled HCP subjects (N=20) with respiratory belt recordings of sufficient quality. This code contains a set of command-line python scripts for replicating the analysis. The code was run with Python 3.11.3.

# Installation
This repository uses `uv` for environment and dependency management. To create the virtual environment and install the project with the locked dependencies, run:
```
uv sync
```

# Exporting requirements.txt
The `requirements.txt` file is generated from `uv.lock`. If you need a pip-compatible export, regenerate it with:
```
uv export --format requirements-txt > requirements.txt
```

# Command-Line Usage
The current analysis entry point is `main.py`. To view the available arguments, run:
```
uv run python main.py --help
```

To replicate the supported analyses, use one of the following commands:

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






