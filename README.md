<div align="center">
  <h2>
    <b>PRISM: Lightweight Long-Term Time Series Forecasting With Period-Based Reorganization and Dual-Axis Convolution</b>
  </h2>
</div>

<div align="center">

![Status](https://img.shields.io/badge/Manuscript-Submitted-orange)
![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch\&logoColor=white)
![Last Commit](https://img.shields.io/github/last-commit/Pigeon1999/PRISM)
![Stars](https://img.shields.io/github/stars/Pigeon1999/PRISM?style=flat)

</div>

This repository provides the PyTorch implementation accompanying our submitted manuscript:

> **PRISM: Lightweight Long-Term Time Series Forecasting With Period-Based Reorganization and Dual-Axis Convolution**

PRISM stands for **Period Reorganization With Intra-Period Sequence Mixing**.

> **Manuscript status:** Submitted and not yet published.
> The repository and experimental results may be updated during the review process.


## 🔍 Overview

PRISM is a lightweight framework for long-term time series forecasting designed for resource-constrained environments.

Existing minimalist forecasting models primarily rely on linear projections. Although computationally efficient, linear projections do not exploit local temporal patterns through shared weights and require parameters proportional to their input and output dimensions.

PRISM addresses this limitation through a **dual-axis convolutional architecture**:

* **Period-based reorganization** transforms a one-dimensional sequence into a two-dimensional periodic representation.
* **Sequence summarization** applies dilated Conv1d across periods to efficiently capture long-range cross-period patterns.
* **Temporal mixing** applies circular-padded Conv1d within each period to learn correlations among temporal phases.
* **Lightweight forecasting** uses a single linear layer to map the extracted representation to the forecasting horizon.
* **ACF-guided configuration** connects the temporal characteristics of each dataset to interpretable hyperparameter choices.

<p align="center">
  <img src="./Figures/Figure1.jpg" alt="Efficiency and accuracy comparison of PRISM" width="78%">
</p>

PRISM establishes a favorable efficiency–accuracy operating point by maintaining competitive forecasting performance with an extremely small parameter count and computational cost.


## 🏗️ Model Architecture

<p align="center">
  <img src="./Figures/Figure2.png" alt="Overall architecture of PRISM" width="100%">
</p>

The PRISM forecasting pipeline consists of five primary stages:

1. **Instance normalization** removes the temporal mean from each input sequence.
2. **2D reorganization** reshapes the sequence according to its known period.
3. **Sequence summarization** applies dilated Conv1d along the cross-period axis.
4. **Temporal mixing** applies circular-padded Conv1d along the intra-period axis.
5. **Linear forecasting and reconstruction** generate and restore the final one-dimensional forecast.

The complete model is implemented in [`models/PRISM.py`](./models/PRISM.py).

### Sequence Summarization

Observations with the same phase across consecutive periods are processed using dilated Conv1d. Kernel size, stride, and dilation jointly determine the receptive field and compression ratio.

This operation captures long-range cross-period dependencies while requiring only a small convolution kernel rather than a dense linear projection.

### Temporal Mixing

Circular-padded Conv1d is applied across the temporal phases within each period.

Circular padding preserves continuity between the final phase of one period and the initial phase of the next period, allowing PRISM to model the cyclic structure of periodic time series without introducing artificial boundary discontinuities.


## 📊 Experimental Results

### Forecasting Performance
PRISM achieves top-three forecasting performance in **24 of 32 evaluation settings**. It ranks first across all forecasting horizons on ETTh1 and maintains top-three performance across all horizons on ETTh2 and ETTm2.

The model also remains competitive on high-dimensional datasets such as Electricity and Traffic while using substantially fewer computational resources than Transformer-based forecasting models.

<p align="center">
  <img src="./Figures/Table2.jpg" alt="MSE comparison of multivariate long-term time series forecasting results" width="100%">
</p>



### Parameter and Computational Efficiency
PRISM achieves top-two performance in parameter count and MACs across most evaluated settings.

Under the representative configuration with an input length of 720 and a forecasting horizon of 96, PRISM requires only **45 parameters** and **13.78K MACs**.

For ETTh1 with a forecasting horizon of 720, PRISM reduces MACs by approximately:

These results demonstrate that PRISM provides balanced efficiency in both model size and computational cost rather than optimizing only one of these dimensions.

<p align="center">
  <img src="./Figures/Table3.jpg" alt="Comparison of parameter counts and MACs" width="100%">
</p>



## 🔬 ACF-Guided Hyperparameter Analysis
PRISM uses the autocorrelation function to interpret the temporal structure of a dataset and guide the selection of its primary hyperparameters:

* `kernel_size` determines how many periods are summarized by each convolution.
* `stride` controls the compression ratio of cross-period features.
* `dilation` determines the receptive field across periods.
* `temporal_kernel_size` controls the range of intra-period temporal interactions.

Datasets with sustained periodicity can benefit from a larger dilation, whereas datasets with rapidly decaying autocorrelation generally favor a more moderate receptive field.

<p align="center">
  <img src="./Figures/Figure3.jpg" alt="Autocorrelation function of the Electricity dataset" width="75%">
</p>

### Dilation Analysis
Increasing dilation expands the cross-period receptive field while reducing MACs. However, an excessively large dilation can skip important local relationships and degrade forecasting accuracy.

On the Electricity dataset, the forecasting error remains stable up to a moderate dilation value before increasing sharply, demonstrating the importance of balancing computational efficiency and temporal coverage.

<p align="center">
  <img src="./Figures/Figure4.jpg" alt="Impact of dilation on forecasting performance and MACs" width="75%">
</p>



## 🔄 Effect of Temporal Mixing
Temporal mixing consistently improves forecasting performance on ETTh1 and provides smaller but meaningful improvements on Traffic.

Its effect is limited on Electricity, whose rapidly decaying intra-period autocorrelation provides less temporal structure for the mixing operation to exploit. Across the evaluated datasets, temporal mixing contributes either positively or neutrally and does not degrade performance.

<p align="center">
  <img src="./Figures/Figure5.jpg" alt="Forecasting performance with and without temporal mixing" width="100%">
</p>


### Temporal Mixing Kernel Size
The optimal temporal mixing kernel size depends on the strength and range of intra-period correlations.

Larger kernels can capture broader phase interactions in strongly periodic data, but they may introduce noise or destructive interactions when intra-period dependencies are weak. This result further supports configuring PRISM according to the temporal characteristics observed through ACF analysis.

<p align="center">
  <img src="./Figures/Figure6.jpg" alt="Impact of temporal mixing kernel size" width="100%">
</p>




## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Pigeon1999/PRISM.git
cd PRISM
```

### 2. Create a Conda environment

```bash
conda create -n prism python=3.8.20 -y
conda activate prism
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


## 📦 Dataset Preparation

The datasets are not included in this repository.

Create a `Dataset` directory in the project root:

```bash
mkdir Dataset
```

Place the required CSV files in the following directory:

```text
PRISM/
└── Dataset/
    └── <dataset>.csv
```

The dataset path, forecasting horizon, and PRISM hyperparameters can be configured in [`main.ipynb`](./main.ipynb).


## 📝 Manuscript Status

This work has been submitted and is not yet published.

A public manuscript link, publication information, and BibTeX citation will be added after publication.
