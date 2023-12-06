OneMod
######

.. image:: ../pipeline.png


In many applications in epidemiology, we need to estimate a single quantity (e.g., incidence or prevalence) by leveraging covariates and correlations across multiple dimensions (e.g., age, location, or year).
This pipeline package estimates quantities using the following approach:

#. Statistical modeling with covariates

   * Create generalized linear models with `RegMod <https://github.com/ihmeuw-msca/regmod>`_

   * Explore covariate combinations with `ModRover <https://github.com/ihmeuw-msca/modrover>`_

#. Coefficient smoothing with `RegMod Smooth <https://github.com/ihmeuw-msca/regmodsm>`_
#. Smoothing across dimensions

   * Smooth predictions using weighted averages with `WeAve <https://github.com/ihmeuw-msca/weighted-average>`_

   * Smooth predictions using mixed-effects models (i.e., `MR-BRT <https://github.com/ihmeuw-msca/mrtool>`_ with `SWiMR <https://hub.ihme.washington.edu/display/MSCA/Similarity-Weighted+Meta-Regression+%28SWiMR%29+models>`_)

#. Ensemble smoothed predictions

#######################
Table of Contents
#######################

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    installation
    usage
    stages/index
    example/index
