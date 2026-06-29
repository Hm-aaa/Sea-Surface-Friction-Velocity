Parameterizing Sea Surface Friction Velocity Under Low-to-Moderate Wind Conditions with Ensemble Learning

This repository provides the source code used in the manuscript:

Parameterizing Sea Surface Friction Velocity Under Low-to-Moderate Wind Conditions with Ensemble Learning

The study develops a data-driven stacking ensemble framework for estimating sea surface friction velocity under low-to-moderate wind conditions. The model is trained and evaluated using field observations from a coastal air–sea flux observation platform in the northern South China Sea. The framework combines wind and wave state predictors to improve the representation of turbulent momentum exchange across the air–sea interface.

1. Overview

Sea surface friction velocity is a key variable for characterizing air–sea momentum exchange and is widely used in bulk flux algorithms, boundary-layer theory, and coupled atmosphere–ocean models. Traditional parameterization schemes often rely on prescribed empirical formulations and may have difficulty representing nonlinear and regime-dependent effects associated with wind forcing and wave state, especially under coastal conditions.

This repository includes scripts for:

preprocessing turbulent wind and wave observations;
constructing wind- and wave-state predictors;
training machine-learning models for friction velocity estimation;
building a stacking ensemble model;
comparing the ensemble model with conventional parameterization schemes;
generating figures and tables used in the manuscript.
2. Study Site and Data

The observational data were collected from a fixed coastal flux observation platform at the Bohe Marine Meteorological Science Experiment Base of the China Meteorological Administration in the northern South China Sea.

The observational system includes:

ultrasonic anemometers for turbulent wind measurements;
wave and current observations from an acoustic Doppler current profiler;
derived wind and wave state variables used as model predictors.

The target variable is the observed sea surface friction velocity estimated from turbulent flux measurements.

Due to data-sharing restrictions, the raw observational data are not included in this repository. Users may organize their own data according to the required input format described below.

3. Predictors

The ensemble framework uses eight wind- and wave-related predictors:

neutral 10 m wind speed;
wave age;
wave steepness;
significant wave height;
wave direction;
peak phase speed;
peak wavenumber;
peak frequency.

The target variable is:

sea surface friction velocity.
4. Model Framework

The proposed framework adopts a two-level stacking architecture.

At the first level, multiple machine-learning algorithms are used as base learners, including tree-based models, kernel-based models, linear models, and neural-network models.

At the second level, a Huber regressor is used as the meta-learner to combine the predictions from the base learners. The Huber regressor improves robustness against outliers and helps stabilize the final ensemble prediction.

The general workflow is:

Raw observations
        ↓
Quality control and 30-min averaging
        ↓
Feature construction
        ↓
Train–test split
        ↓
Base learner training
        ↓
Stacking ensemble training
        ↓
Model evaluation
        ↓
Comparison with conventional parameterizations
        ↓
Figure and table generation
