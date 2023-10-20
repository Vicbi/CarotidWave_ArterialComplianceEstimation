c# CarotidWave_ArterialComplianceEstimation
## On the assessment of arterial compliance from carotid pressure waveform

This repository contains the Python code for random forest and artificial neural network-enabled estimators for total arterial compliance. Carotid blood pressure waveform is used as input.

This code contains an enhanced iteration of the code and analysis employed in this study. It is important to note that the final results may exhibit variances from those originally reported in the paper. These discrepancies arise primarily from refinements in how we process the input waveform, add the artificial noise and compute the associated features. These updates have been implemented to ensure a more precise and comprehensive analysis, while they do not lead to conflicting results in comparison to the original findings.

**Abstract**

In a progressively aging population, it is of utmost importance to develop reliable, noninvasive, and cost-effective tools to estimate biomarkers that can be indicative of cardiovascular risk. Various pathophysiological conditions are associated to
changes in the total arterial compliance (CT), and thus, its estimation via an accurate and simple method is valuable. Direct
noninvasive measurement of CT is not feasible in the clinical practice. Previous methods exist for indirect estimation of CT,
which, however, require noninvasive, yet complex and expensive, recordings of the central pressure and flow. Here, we introduce
a novel, noninvasive method for estimating CT from a single carotid waveform measurement using regression analysis. Features were extracted from the carotid wave and were combined with demographic data. A prediction pipeline was adopted for estimating CT using, first, a feature-based regression analysis and, second, the raw carotid pulse wave. The proposed methodology was appraised using the large human cohort (N = 2,256) of the Asklepios study. Accurate estimates of CT were yielded for both prediction schemes, namely, r = 0.83 and normalized root mean square error (nRMSE) = 9.58% for the feature-based model, and r = 0.83 and nRSME = 9.67% for the model that used the raw signal. The major advantage of this method pertains to the simplification of the technique offering easily applicable and convenient CT monitoring. Such an approach could offer promising applications, ranging from fast and cost-efficient hemodynamical monitoring by the physician to integration in wearable technologies.

<img width="751" alt="Screenshot at Oct 19 18-20-40" src="https://github.com/Vicbi/CarotidWave_ArterialComplianceEstimation/assets/10075123/bee522b9-39c8-45fb-9b1f-3f66081b6636">


**Original Publication**

For a comprehensive understanding of the methodology and background, refer to the original publication: Bikia, V., Segers, P., Rovas, G., Pagoulatou, S., & Stergiopulos, N. (2021). On the assessment of arterial compliance from carotid pressure waveform. American Journal of Physiology-Heart and Circulatory Physiology, 321(2), H424-H434.

**Citation**

If you use this code in your research, please cite the original publication:

Bikia, V., Segers, P., Rovas, G., Pagoulatou, S., & Stergiopulos, N. (2021). On the assessment of arterial compliance from carotid pressure waveform. American Journal of Physiology-Heart and Circulatory Physiology, 321(2), H424-H434. https://doi.org/10.1152/ajpheart.00241.2021

**License**

This project is licensed under the Apache License 2.0 - see the LICENSE.md file for details.

This work was developed as part of a research project undertaken by the Laboratory of Hemodynamics and Cardiovascular Technology at EPFL (https://www.epfl.ch/labs/lhtc/).


Feel free to reach out at vickybikia@gmail.com if you have any questions or need further assistance!
