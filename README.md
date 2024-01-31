# EnsembleXAI
<!-- <hr/> -->

<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?logo=github&style=flat&color=green)](./LICENSE)
[![Docs - GitHub.io](https://img.shields.io/static/v1?style=flat&color=pink&label=docs&message=EnsembleXAI)](./docs/_build/html/index.html)
![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-grey.svg?logo=python&logoColor=blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-in%20progress-b31b1b.svg)](https://arxiv.org/abs/TODO)
<!--- BADGES: END --->

This repository presents the EnsembleXAI library, a comprehensive Python library for ensembling explainable artificial intelligence (XAI) methods. The EnsembleXAI library is designed to be user-friendly and provides various functionalities for ensembling XAI methods, evaluating them, and integrating them into your deep learning pipelines.

## About XAI ensembles

### NormEnsembleXAI

The NormEnsembleXAI method is designed to handle the diverse value ranges produced by various explanation methods. To address this, the algorithm employs normalization techniques, including Second Moment Scaling, Normal Standardization, or Robust Standardization. Subsequently, it utilizes aggregation functions for ensembling explanations, including Maximum, Minimum, and Mean.

![NormEnsembleXAI method](./docs/images/normensembleXAI.png)

### SupervisedXAI

In SupervisedXAI method, explanations for an instance are reshaped and concatenated into a matrix, which constitutes the training dataset X. The segmentation mask is similarly reshaped into a one-dimensional vector, serving as a set of labels Y. A multioutput Kernel Ridge Regression (KRR) model is then trained to predict the values of Y, using the explanations transformed into the X matrix as input.

![SupervisedXAI method](./docs/images/supervisedXAI.png)

### Autoweighted

In Autoweighted method, explanations are assessed using a chosen metric, and subsequently, an Ensemble Score (ES) is computed for each explanation method. The final XAI ensemble  is constructed as a weighted mean of normalized explanations, with individual weights determined by their respective ES values.

![Autoweighted method](./docs/images/autoweighted.png)

#### References of Algorithms


* `NormEnsembleXAI`: [Hryniewska-Guzik, W., Sawicki, B., & Biecek, P. (2024). NormEnsembleXAI: Unveiling the Strengths and Weaknesses of XAI Ensemble Techniques.](https://arxiv.org/abs/2401.17200)
* `SupervisedXAI`: [Zou, L., Goh, H. L., Liew, C. J. Y., Quah, J. L., Gu, G. T., Chew, J. J., Prem Kumar, M., Ang, C. G. L., & Ta, A. (2022). Ensemble image explainable AI (XAI) algorithm for severe community-acquired pneumonia and COVID-19 respiratory infections. IEEE Transactions on Artificial Intelligence, 1–1.](https://doi.org/10.1109/TAI.2022.3153754)
* `Autoweighted`: [Bobek, S., Bałaga, P., & Nalepa, G. J. (2021). Towards Model-Agnostic Ensemble Explanations. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 12745 LNCS, 39–51.](https://doi.org/10.1007/978-3-030-77970-2_4)

## Installation

**Installation Requirements**
- numpy >= 1.24.4
- scikit-learn >= 1.1.2
- torch >= 2.1.0

To install the package use pip. Having build the package, just run the following command from main project directory.

```bash
 pip install dist/EnsembleXAI-0.0.1.tar.gz
 ```

If you have access to project GitHub repository you can download, build and install the package via command:

```bash
pip install git+https://github.com/anonymous-conference-journal/EnsembleXAI.git
```

## Modules

The EnsembleXAI library consists of three main modules:

### Ensemble

The Ensemble module provides functionalities for ensembling explanations from various XAI methods. It allows you to aggregate multiple explanations using methods such as Maximum, Minimum, and Mean. Here's a quick example of how to use the Ensemble module:

```python3
import torch
from EnsembleXAI.Ensemble import normEnsembleXAI
from captum.attr import IntegratedGradients, GradientShap, Saliency

net = ImageClassifier()
inputs = torch.randn(1, 3, 32, 32)

ig = IntegratedGradients(net).attribute(inputs, target=3)
gs = GradientShap(net).attribute(inputs, target=3)
sal = Saliency(net).attribute(inputs, target=3)

explanations = torch.stack([ig, gs, sal], dim=1)
agg = normEnsembleXAI(explanations, aggregating_func='avg')
```

### Metrics
The Metrics module contains various explainability metrics, primarily those introduced in~Zou et al. and Bobek et al.. These metrics allow you to evaluate the quality and reliability of the ensembled explanations. Here's an example of how to use the Metrics module:

```python3
import torch
from EnsembleXAI.Metrics import consistency

# Calculate consistency for two explanations
con = consistency(explanation1, explanation2)

print(f"Consistency: {con}")
```

### Normalization
The Normalization module provides scaling and standardizing functions to handle the diverse value ranges produced by different explanation methods. You can use these functions to preprocess your explanations before ensembling. Here's an example of how to use the Normalization module:

```python3
from EnsembleXAI.Normalization import second_moment_normalize

# Scale an explanation using Second Moment Scaling
scaled_explanation = second_moment_normalize(explanation)
```

## Documentation

The complete documentation for EnsembleXAI is available online [EnsembleXAI Documentation](https://anonymous-conference-journal.github.io/EnsembleXAI/).

## License

This project is licensed under the BSD License. See the [LICENSE](LICENSE) file for details.
