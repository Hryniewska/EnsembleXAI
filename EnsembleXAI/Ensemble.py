import torch

from copy import deepcopy
from EnsembleXAI.Metrics import ensemble_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from torch import Tensor, stack
from typing import TypeVar, Tuple, List, Callable, Union, Any

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...])


def _apply_over_axis(x: torch.Tensor, function: Callable, axis: int) -> torch.Tensor:
    """
    Apply function over axis in tensor.

    Reduces input by one dimension.

    Parameters
    ----------
    x : Tensor
        Tensor to apply function on.
    function: Callable
        Function to apply.
    axis: int
        Axis to apply function over.
    Returns
    -------
    Tensor
        Result of a function call.
    """
    return torch.stack([
        function(x_i) for i, x_i in enumerate(torch.unbind(x, dim=axis), 0)
    ], dim=axis)


def _reformat_input_tensors(inputs: TensorOrTupleOfTensorsGeneric) -> Tensor:
    """
    Convert input into unified tensor.

    Dimensions of the output tensor correspond to observations, explanations for one observation,
    channels, height, width of single observation.

    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Tensor, list or tuple of tensors.
    Returns
    -------
    Tensor
        Tensor with unified dimensions.
    """
    parsed_inputs = deepcopy(inputs)
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        if inputs[0].dim() <= 4:
            # multiple observations with explanations as tensor
            parsed_inputs = stack(inputs)
    if parsed_inputs.dim() == 3:
        parsed_inputs = parsed_inputs[None, :]
    if parsed_inputs.dim() == 4:
        # single observation with multiple explanations
        parsed_inputs = parsed_inputs[None, :]

    return parsed_inputs


def normEnsembleXAI(inputs: TensorOrTupleOfTensorsGeneric,
                    aggregating_func: Union[str, Callable[[Tensor], Tensor]]) -> Tensor:
    """
    Aggregate explanations in the simplest way.

    Use provided aggregating functions or pass a custom callable. Combine explanations for every observation and get
    one aggregated explanation for every observation.

    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    aggregating_func : Union[str, Callable[[Tensor], Tensor]]
        Aggregating function. Can be string, one of 'avg', 'min', 'max',
        or a function from a list of tensors to tensor.

    Returns
    -------
    Tensor
        Aggregated explanations. Dimensions correspond to no. of observations, aggregated explanation.

    See Also
    --------
    autoweighted : Aggregation weighted by quality of each explanation.
    supervisedXAI : Use Kernel Ridge Regression for aggregation, suitable when masks are available.

    Examples
    --------

    >>> import torch
    >>> from EnsembleXAI.Ensemble import normEnsembleXAI
    >>> from captum.attr import IntegratedGradients, GradientShap, Saliency
    >>> net = ImageClassifier()
    >>> inputs = torch.randn(1, 3, 32, 32)
    >>> ig = IntegratedGradients(net).attribute(inputs, target=3)
    >>> gs = GradientShap(net).attribute(inputs, target=3)
    >>> sal = Saliency(net).attribute(inputs, target=3)
    >>> explanations = torch.stack([ig, gs, sal], dim=1)
    >>> agg = normEnsembleXAI(explanations, 'avg')

    """
    # input tensor dims: observations x explanations x single explanation
    assert isinstance(aggregating_func, str) or isinstance(
        aggregating_func, Callable)

    if isinstance(aggregating_func, str):
        assert aggregating_func in ['avg', 'min', 'max', 'max_abs']

    parsed_inputs = _reformat_input_tensors(inputs)

    input_size = parsed_inputs.size()
    n_explanations = input_size[1]
    new_size = (input_size[0], 1, input_size[2], input_size[3], input_size[4])

    if aggregating_func == 'avg':
        output = torch.squeeze(1 / n_explanations *
                               parsed_inputs.sum_to_size(new_size), dim=1)

    if aggregating_func == 'max':
        output = parsed_inputs.amax(1)

    if aggregating_func == 'min':
        output = parsed_inputs.amin(1)

    if aggregating_func == 'max_abs':
        output = torch.abs(parsed_inputs).amax(1)

    if isinstance(aggregating_func, Callable):
        output = _apply_over_axis(parsed_inputs, aggregating_func, 0)

    return output


def _normalize_across_dataset(parsed_inputs: Tensor, delta=0.00001):
    """
    Mean, variance normalization across all data in inputs.

    When comparing different explanations it is mandatory to normalize values, since every XAI algorithm
    has its own set of return values.

    Parameters
    ----------
    parsed_inputs : Tensor
        Inputs to be normalized in parsed form.
    delta : Float
        Minimal permitted variance. If variance is smaller than delta raises ZeroDivisionError.
    Returns
    -------
    Tensor
        Normalized explanations.
    """
    var, mean = torch.var_mean(
        parsed_inputs, dim=[0, 2, 3, 4], unbiased=True, keepdim=True)
    if torch.min(var.abs()) < delta:
        raise ZeroDivisionError("Variance close to 0. Can't normalize")
    return (parsed_inputs - mean) / torch.sqrt(var)


def autoweighted(inputs: TensorOrTupleOfTensorsGeneric,
                 metric_weights: List[float],
                 metrics: Union[List[Callable], None] = None,
                 precomputed_metrics: Union[Any, np.ndarray, torch.Tensor] = None) -> Tensor:
    """
    Aggregate explanations weighted by their quality measured by metrics.

    This function in an implementation of explanation ensemble algorithm published in [1]_. It uses
    :func:`EnsembleXAI.Metrics.ensemble_score` to calculate quality of each explanation. One of `metrics`
    or `precomputed_metrics` should be passed.

    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    metrics : List[Callable], default None
        Metrics used to assess the quality of an explanation. Ignored when precomputed_metrics is not None.
    metric_weights : List[float]
        Weights used to calculate :func:`EnsembleXAI.Metrics.ensemble_score` of every explanation.
    precomputed_metrics: Any, default None
        Metrics' values can be precomputed and passed as an argument. Need to be in 3 dimensional format
        where dimensions correspond to observations, explanations and metrics.
        Supported formats are numpy ndarray and torch tensor.

    Returns
    -------
    Tensor
        Weighted arithmetic mean of explanations, weighted by :func:`EnsembleXAI.Metrics.ensemble_score`.
        Dimensions correspond to no. of observations, aggregated explanation.

    See Also
    --------
    normEnsembleXAI : Simple aggregation by function, like average.
    supervisedXAI : Use Kernel Ridge Regression for aggregation, suitable when masks are available.

    Notes
    -----
    Explanations are normalized by mean and standard deviation before aggregation to ensure comparable values.

    References
    ----------
    .. [1] Bobek, S., Bałaga, P., Nalepa, G.J. (2021), "Towards Model-Agnostic Ensemble Explanations."
        In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds)
        Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12745. Springer,
        Cham. https://doi.org/10.1007/978-3-030-77970-2_4

    Examples
    --------
    >>> import torch
    >>> from EnsembleXAI.Ensemble import autoweighted
    # We have a tensor of 4 explanations for 15 observations,
    # each with 3 channels and image size 32 x 32
    >>> explanations = torch.randn(15, 4, 3, 32, 32)
    # We use precomputed metrics
    # 2 metrics to evaluate each of 4 explanation for 15 observations
    >>> metrics = torch.rand(size=(15, 4, 2))
    >>> ensembled_explanations = autoweighted(explanations,
    ...                                       metric_weights=[0.2, 0.8],
    ...                                       precomputed_metrics=metrics)

    """

    assert precomputed_metrics is not None or metrics is not None, "one of metrics or precomputed_metrics must not be None"
    parsed_inputs = _reformat_input_tensors(inputs)
    if precomputed_metrics is None:
        # calculate metrics
        metric_vals = [
            [metric(explanation) for metric in metrics] for explanation in torch.unbind(parsed_inputs, dim=1)]
    else:
        if isinstance(precomputed_metrics, np.ndarray):
            metric_vals = torch.from_numpy(precomputed_metrics)
        elif isinstance(precomputed_metrics, torch.Tensor):
            metric_vals = precomputed_metrics
        else:
            raise ValueError(
                "precomputed_metrics should be numpy ndarray or torch tensor")
        assert metric_vals.dim() == 3, "precomputed_metrics should have 3 dims"
        metric_vals = [y.unbind(0) for y in metric_vals.transpose(
            2, 0).transpose(1, 0).unbind(0)]
    ensemble_scores = torch.stack(
        [ensemble_score(metric_weights, metric_val) for metric_val in metric_vals])
    ensemble_scores.transpose_(0, 1)

    normalized_exp = _normalize_across_dataset(parsed_inputs)

    # allocate array for ensemble explanations
    n = parsed_inputs.size()[0]
    results = [0] * n

    for observation, scores, i in zip(torch.unbind(normalized_exp), torch.unbind(ensemble_scores), range(n)):
        # multiply single explanation by its ensemble score
        weighted_exp = torch.stack([
            exp * score for exp, score in zip(torch.unbind(observation), torch.unbind(scores))
        ])
        # sum weighted explanations and normalize by sum of scores
        ensemble_exp = torch.sum(weighted_exp, dim=0) / scores.sum()
        results[i] = ensemble_exp

    return torch.stack(results)


def _auto_calculate_weights(masks: np.ndarray) -> np.ndarray:
    size = masks.shape[1]
    non_zero = np.count_nonzero(masks, axis=1)
    return size / non_zero


def supervisedXAI(inputs: TensorOrTupleOfTensorsGeneric, masks: TensorOrTupleOfTensorsGeneric, n_folds: int = 3,
                  weights: Union[str, TensorOrTupleOfTensorsGeneric,
                                 np.ndarray, None] = None,
                  shuffle=False, random_state=None) -> Tensor:
    """
    Aggregate explanations by training supervised machine learning model.

    This function in an implementation of explanation ensemble algorithm published in [1]_. It uses
    :class:`sklearn.kernel_ridge.KernelRidge` to train the Kernel Ridge Regression (KRR) model with explanations
    as inputs :math:`X` and masks as output :math:`y`.
    K-Fold split is used to generate aggregated explanations without information leakage. Internally uses
    :class:`sklearn.model_selection.KFold` to make the split.


    Parameters
    ----------
    inputs : TensorOrTupleOfTensorsGeneric
        Explanations in form of tuple of tensors or tensor. `inputs` dimensions correspond to no. of observations,
        no. of explanations for each observation, and single explanation.
    masks : TensorOrTupleOfTensorsGeneric
        Masks used by KRR model as output. Should be 3 dimensional shape, where dimensions correspond to no. of observations,
        and single mask. Size of single mask should be the same as size of single explanation in `inputs`.
    n_folds : int, default 3
        Number of folds used to train the KRR model. `n_folds` should be an `int` greater than 1. When `n_folds` is
        equal to no. of observations in `inputs`, "leave one out" training is done.
    weights : Union[str, TensorOrTupleOfTensorsGeneric, np.ndarray, None], default None
        Sample weights for training the KRR. If None, weights are uniform. If 'auto',
        weight of an observation is inversely proportional to the area of the observation's mask. Can be also provided
        as tensor, list, tuple or numpy array of custom values. Weights can be used to promote smaller masks.
    shuffle: Any, default False
        If `True` inputs and masks will be shuffled before k-fold split. Internally passed
        to :class:`sklearn.model_selection.KFold`.
    random_state: Any, default None
        Used only when `shuffle` is `True`. Internally passed to :class:`sklearn.model_selection.KFold`.
    Returns
    -------
    Tensor
        Tensor of KRR model outputs, which are the aggregated explanations. It has 3 dimensions.

    See Also
    --------
    normEnsembleXAI : Simple aggregation by function, like average.
    autoweighted : Aggregation weighted by quality of each explanation.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> from EnsembleXAI.Ensemble import normEnsembleXAI
    >>> from captum.attr import IntegratedGradients, GradientShap, Saliency
    >>> net = ImageClassifier()
    >>> input = torch.randn(15, 3, 32, 32)
    >>> masks = torch.randint(low=0, high=2, size=(15, 32, 32))
    >>> ig = IntegratedGradients(net).attribute(input, target=3)
    >>> gs = GradientShap(net).attribute(input, target=3)
    >>> sal = Saliency(net).attribute(input, target=3)
    >>> explanations = torch.stack([ig, gs, sal], dim=1)
    >>> krr_explanations = supervisedXAI(explanations, masks)
    """

    assert n_folds > 1
    # reshape do 1d array for each observation

    parsed_inputs = _reformat_input_tensors(inputs)
    input_shape = parsed_inputs.shape
    numpy_inputs = parsed_inputs.numpy().reshape((len(inputs), -1))
    labels = _reformat_input_tensors(masks).squeeze(
    ).numpy().reshape((len(parsed_inputs), -1))
    if isinstance(weights, str):
        assert weights == 'auto'
        weights = _auto_calculate_weights(labels)
    assert len(parsed_inputs) == len(
        masks), "Inconsistent number of observations in masks and inputs"
    assert len(
        parsed_inputs) >= n_folds, "Number of observations should be greater or equal than number of folds"
    if weights is not None:
        assert len(weights) == len(masks)
    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)

    ensembled = [0] * n_folds
    indices = np.empty(1, dtype=int)

    for idx, (train_index, test_index) in enumerate(kf.split(numpy_inputs, labels)):
        # get observations split by k-fold
        X_train, X_test = (
            numpy_inputs[train_index]), (numpy_inputs[test_index])
        y_train = labels[train_index]
        if weights is not None:
            iter_weights = weights[train_index]
        else:
            iter_weights = None
        # train KRR
        krr = KernelRidge(
            alpha=1,  # regularization
            kernel='polynomial'  # choose one from:
            # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/pairwise.py#L2050
        )
        krr.fit(X_train, y_train, sample_weight=iter_weights)
        # predict masks for observations currently in test group
        ensembled[idx] = krr.predict(X_test).reshape(
            (tuple([len(X_test)]) + input_shape[3:5]))
        # reshape predictions and save them and indices to recreate original order later
        indices = np.concatenate([indices, test_index])

    # sort output to match input order
    indices = indices[1:]
    ensembled = np.concatenate(ensembled)
    ensembled_ind = indices.argsort()
    ensembled = ensembled[ensembled_ind[::1]]

    return torch.from_numpy(ensembled)
