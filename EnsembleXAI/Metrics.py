from typing import Callable, Union, List, Tuple
import itertools

import numpy as np
import torch


def replace_masks(
        images: torch.Tensor, replacement_index: torch.BoolTensor, value: Union[int, float] = 0
) -> torch.Tensor:
    """
    Replaces values in Tensor indexed by a boolean tensor.

    Replaces data in the Tensor with one value in the spots where boolean index Tensor is True.
    In the case when a 4D tensor with a 3D index is given, index is repeated along the second dimension to fit the data shape.

    Parameters
    ----------
    images: torch.Tensor
        Tensor of any shape, in most cases 4D Tensor of the images with shape (number of photos, RGB channel, height, width)
    replacement_index: torch.BoolTensor
        Boolean Tensor of shape same as images or in case of the 4D images Tensor, a
        3D boolean Tensor where true corresponds index to be replaced with shape (number of photos, height, width)
    value: int or float
        Value to use for replacing the data with.

    Returns
    -------
    torch.Tensor
        Tensor of same shape as input with the replaced data.

    See Also
    --------
    decision_impact_ratio: Measures the average number of changes in the predictions after hiding the critical area.
    confidence_impact_ratio: Measures the average change in probabilities after hiding the critical area.

    Examples
    --------
    >>> import torch
    >>> image = torch.ones([3,3])
    >>> image
    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])
    >>> index = torch.BoolTensor([False, True, False]).repeat(3,1)
    >>> index
    tensor([[False,  True, False],
            [False,  True, False],
            [False,  True, False]])
    >>> replace_masks(image, index, 0)
    tensor([[1., 0., 1.],
            [1., 0., 1.],
            [1., 0., 1.]])
    >>> image_4D = torch.ones([1,3,4,4])
    >>> image[0,0]
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    >>> index_3D = torch.BoolTensor([False, True, True, False]).repeat(1,4,1)
    >>> index_3D
    tensor([[[False,  True,  True, False],
             [False,  True,  True, False],
             [False,  True,  True, False],
             [False,  True,  True, False]]])
    >>> replaced_image = replace_masks(image_4D, index_3D, 2)
    >>> replaced_image[0,0]
    tensor([[1., 2., 2., 1.],
            [1., 2., 2., 1.],
            [1., 2., 2., 1.],
            [1., 2., 2., 1.]])
    """
    temp_images = torch.clone(images)
    # 3D Tensor needs to be reshaped over RGB channel for the indexing of images
    if len(replacement_index.shape) == 3 and len(temp_images.shape) == 4:
        replacement_index = replacement_index.unsqueeze(dim=1).repeat(1, temp_images.shape[1], 1, 1)
    temp_images[replacement_index] = value
    return temp_images


def tensor_to_list_tensors(tensors: torch.Tensor, depth: int) -> List[torch.Tensor]:
    """
    Splits first n dimensions of a Tensor into a list of Tensors.

    Splits the first n Tensor dimensions into a list of Tensors of length equal to product of the split dimensions sizes.
    Resulting Tensors have dimensions reduced by a factor of n.

    Parameters
    ----------
    tensors: torch.Tensor
        Tensor to be split into a list. Number of dimensions greater than depth parameter.
    depth: int
        Value representing the depth to which to split the tensors, starting from the first dimension.
        Therefore, depth=1 represents splitting only the first dimension. Thus depth cannot be larger than the length of the Tensors shape.

    Returns
    -------
    list of torch.Tensor
        A single list consisting of all the split Tensors.

    See Also
    --------
    consistency: Metric representing how similar are different explanations of one photo.
    stability: Measures how similar/stable are explanations of similar photos.

    Examples
    --------
    >>> import torch
    >>> dim1 = torch.stack([torch.Tensor([1, 1, 1, 1]), torch.Tensor([2, 2, 2, 2])])
    >>> dim2 = torch.stack([torch.Tensor([3, 3, 3, 3]), torch.Tensor([4, 4, 4, 4])])
    >>> stacked_tensor = torch.stack([dim1, dim2])
    >>> stacked_tensor
    tensor([[[1., 1., 1., 1.],
             [2., 2., 2., 2.]],
            [[3., 3., 3., 3.],
             [4., 4., 4., 4.]]])
    >>> tensor_to_list_tensors(stacked_tensor, depth=1)
    [tensor([[1., 1., 1., 1.],
             [2., 2., 2., 2.]]),
    tensor([[3., 3., 3., 3.],
            [4., 4., 4., 4.]])]
    >>> tensor_to_list_tensors(stacked_tensor, depth=2)
    [tensor([1., 1., 1., 1.]),
     tensor([2., 2., 2., 2.]),
     tensor([3., 3., 3., 3.]),
     tensor([4., 4., 4., 4.])]
    """
    # squeezing couses returned tensors to have reduced dimensions
    tensor_list = [
        x.squeeze() for x in torch.tensor_split(tensors, tensors.shape[0], dim=0)
    ]
    for i in range(depth - 1):
        tensor_list = [
            y.squeeze()
            for x in tensor_list
            for y in torch.tensor_split(x, x.shape[0], dim=0)
        ]
    return tensor_list


def matrix_2_norm(
        matrix1: torch.Tensor, matrix2: torch.Tensor, sum_dim: int = None
) -> torch.Tensor:
    """
    Computes the 2-norm of two matrices.

    Computes the 2-norm of two matrices. By default works on the last two dimensions of the Tensor,
    which can be extended by the sum_dim parameter to one of the remaining dimensions of the Tensor.

    Parameters
    ----------
    matrix1: torch.Tensor
        Tensor with one of the matrices to compute the norm.
    matrix2: torch.Tensor
        Tensor with the second of the matrices to compute the norm. Shape has to be either equal to the first matrix,
        only the first dimension of the first matrix can be omitted.
    sum_dim: int
        Optional dimension to extend the calculation to. Indexed as in the original matrix,
        therefore supports both positive and negative indexing.

    Returns
    -------
    torch.Tensor
        Tensor with value or values of the 2-norm. The shape is similar to both of the input matrices,
        except for last two removed dimensions and the optional dimension specified in sum_dim parameter.

    See Also
    --------
    consistency: Metric representing how similar are different explanations of one photo.
    stability: Measures how similar/stable are explanations of similar photos.

    Examples
    --------
    >>> import torch
    >>> onez_2D = torch.ones([3, 3])
    >>> zeroz_2D = torch.zeros([3, 3])
    >>> matrix_2_norm(onez_2D, zeroz_2D)
    tensor(3.)
    >>> onez_3D = torch.ones([4, 3, 2])
    >>> zeroz_3D = torch.zeros([4, 3, 2])
    >>> matrix_2_norm(onez_3D, zeroz_3D)
    tensor([2.4495, 2.4495, 2.4495, 2.4495])
    >>> matrix_2_norm(onez_3D, zeroz_3D, sum_dim=0)
    tensor(4.8990)
    >>> onez_4D = torch.ones([5, 4, 3, 2])
    >>> zeroz_4D = torch.zeros([5, 4, 3, 2])
    >>> matrix_2_norm(onez_4D, zeroz_4D, sum_dim=0)
    tensor([5.4772, 5.4772, 5.4772, 5.4772])
    >>> matrix_2_norm(onez_4D, zeroz_4D, sum_dim=1)
    tensor([4.8990, 4.8990, 4.8990, 4.8990, 4.8990])
    """
    if sum_dim is not None and sum_dim < 0:
        sum_dim = sum_dim + 2
    difference = (matrix1 - matrix2).float()
    norm = torch.linalg.matrix_norm(difference, ord=2)
    # manual extension of the norm calculation to the sum_dim dimension
    if sum_dim is not None:
        norm = torch.pow(norm, 2)
        norm = torch.sum(norm, dim=sum_dim)
        norm = torch.sqrt(norm)
    return norm


def intersection_mask(
        tensor1: torch.Tensor, tensor2: torch.Tensor,
        threshold1: float = 0.0, threshold2: float = 0.0,
        absolute_value: bool = False
) -> torch.Tensor:
    """
    Calculates the intersection of two masks.

    Calculates the logical 'and' intersections of two n-dimensional masks
    where the absolute values of data are greater than the thresholds.

    Parameters
    ----------
    tensor1: torch.Tensor
        First of the two masks.
    tensor2: torch.Tensor
        Second of the two masks.
    threshold1: float
        Threshold value for the first mask.
    threshold2: float
        Threshold value for the second mask.
    absolute_value: bool
        Boolean deciding if the compared tensors should be taken as absolute value.

    Returns
    -------
    torch.Tensor
        Boolean Tensor with True values where the masks intersect with values over the thresholds.

    See Also
    --------
    accordance_recall: Measures how much area of the mask has the explanation covered.
    accordance_precision: Measures how much area of the explanation is covered by the mask.
    intersection_over_union: Measures the average division of intersection area over the union area.
    union_mask: Calculates the union of two masks.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> cross_2d
    tensor([[1., 0., 1.],
            [0., 1., 0.],
            [1., 0., 1.]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> plus_2d
    tensor([[0., 1., 0.],
            [1., 1., 1.],
            [0., 1., 0.]])
    >>> intersection_mask(cross_2d, plus_2d)
    tensor([[False, False, False],
            [False,  True, False],
            [False, False, False]])
    >>> cross_2d_small = 0.4*cross_2d
    >>> plus_2d_small = 0.7*plus_2d
    >>> intersection_mask(cross_2d_small, plus_2d_small, threshold1=0.5)
    tensor([[False, False, False],
            [False, False, False],
            [False, False, False]])
    """
    if absolute_value:
        tensor1 = torch.abs(tensor1)
        tensor2 = torch.abs(tensor2)
    logical_mask = torch.logical_and(
        tensor1 > threshold1, tensor2 > threshold2
    )
    return logical_mask


def union_mask(
        tensor1: torch.Tensor, tensor2: torch.Tensor,
        threshold1: float = 0.0, threshold2: float = 0.0,
        absolute_value: bool = False
) -> torch.Tensor:
    """
    Calculates the union of two masks.

    Calculates the logical 'or' union of two n-dimensional masks where the absolute
    values of data are greater than the thresholds.

    Parameters
    ----------
    tensor1: torch.Tensor
        First of the two masks.
    tensor2: torch.Tensor
        Second of the two masks.
    threshold1: float
        Threshold value for the first mask.
    threshold2: float
        Threshold value for the second mask.
    absolute_value: bool
        Boolean deciding if the compared tensors should be taken as absolute value.

    Returns
    -------
    torch.Tensor
        Boolean Tensor with True values on the union of the masks, where values are over thresholds.

    See Also
    --------
    intersection_over_union: Measures the average division of intersection area over the union area.
    intersection_mask: Calculates the intersection of two masks.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> cross_2d
    tensor([[1., 0., 1.],
            [0., 1., 0.],
            [1., 0., 1.]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> plus_2d
    tensor([[0., 1., 0.],
            [1., 1., 1.],
            [0., 1., 0.]])
    >>> union_mask(cross_2d, plus_2d)
    tensor([[True, True, True],
            [True, True, True],
            [True, True, True]])
    >>> cross_2d_small = 0.4*cross_2d
    >>> plus_2d_small = 0.7*plus_2d
    >>> union_mask(cross_2d_small, plus_2d_small, threshold1=0.5)
    tensor([[False,  True, False],
        [ True,  True,  True],
        [False,  True, False]])
    >>> union_mask(cross_2d_small, plus_2d_small, threshold1=0.0, threshold2=0.8)
    tensor([[ True, False,  True],
            [False,  True, False],
            [ True, False,  True]])
    """
    if absolute_value:
        tensor1 = torch.abs(tensor1)
        tensor2 = torch.abs(tensor2)
    logical_mask = torch.logical_or(
        tensor1 > threshold1, tensor2 > threshold2
    )
    return logical_mask


def consistency_image(models: List, transforms: List, image: torch.tensor, explanator: Callable, **kwargs) -> float:
    """
    Calculate the consistency of explanations across multiple models.

    Calculate the consistency of explanations obtained from different models applied to
    transformed versions of an input image.

    Parameters
    ----------
    models: List[nn.Module]
        List of machine learning models.
    transforms: List[Callable]
        List of image transformation functions corresponding to each model.
    image: torch.Tensor
        Input image as a torch tensor.
    explanator: Callable
        A callable that generates explanations for a given model and input.
    **kwargs
        Additional keyword arguments to be passed to the `explanator` function.

    Returns
    -------
    float
        Consistency value representing the degree of agreement among explanations.

    See Also
    --------
    consistency: Calculate the consistency value of a list of explanations.
    """
    explanations_list = []
    for model, transform in zip(models, transforms):
        image_transformed = transform(image)
        explanations_list += [explanator(image_transformed, model, **kwargs)]

    explanations = torch.cat(explanations_list)
    consistency_value = consistency(explanations)
    return consistency_value


def consistency(explanations: torch.Tensor) -> float:
    """
    Metric representing how similar are different explanations of one photo.

    Metric representing how much do different explanations for the same model or same explanation for different models diverge.
    Maximal value of 1 represents identical explanations and values close to 0 represent greatly differing explanations.
    Metric is calculated as proposed in [1]_.

    Parameters
    ----------
    explanations: torch.Tensor
        Explanations Tensor for the single image. Therefore the required shape is (n, channels, width, height),
        where n stands for the number of explanations and channels stands for a depth of the image (RGB channel in most cases).

    Returns
    -------
    float
        Value of the consistency metric for the input explanations.

    See Also
    --------
    stability: Measures how similar/stable are explanations of similar photos.
    tensor_to_list_tensors: Splits first n dimensions of a Tensor into a list of Tensors.
    matrix_2_norm: Computes the 2-norm of two matrices.

    References
    ----------
    .. [1] Bobek, S., Bałaga, P., Nalepa, G.J. (2021), "Towards Model-Agnostic Ensemble Explanations."
        In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds)
        Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12745. Springer,
        Cham. https://doi.org/10.1007/978-3-030-77970-2_4

    Examples
    --------
    >>> import torch
    >>> ones = torch.ones([3,5,5])
    >>> halfs = 0.5*torch.ones([3,5,5])
    >>> stacked = torch.stack([ones, halfs])
    >>> consistency(stacked)
    0.18761281669139862
    >>> ones2 = torch.ones([4,3,5,5])
    >>> consistency(ones2)
    1.0
    """
    explanations_list = tensor_to_list_tensors(explanations, depth=1)
    diffs = [
        matrix_2_norm(exp1, exp2, sum_dim=0)
        for exp1, exp2 in itertools.combinations(explanations_list, 2)
    ]
    return (1 / (max(diffs) + 1)).item()


def stability_image(explanator: Callable[..., torch.Tensor],
                    image: torch.Tensor,
                    transform: Callable = lambda x: x,
                    n_samples: int = 5,
                    sigma: Union[str, float] = 'auto',
                    random_seed: Union[None, int] = None,
                    return_noised_images: bool = False,
                    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    **kwargs):
    """
    Measure the stability of explanations across similar images.

    Calculate the stability of explanations obtained from a single image and a set of
    similar images generated by adding noise.

    Parameters
    ----------
    explanator: Callable[..., torch.Tensor]
        A callable that generates explanations for an image.
    image: torch.Tensor
        3D tensor of the reference image.
    transform: Callable, optional
        A function for image transformation, by default the identity function.
    n_samples: int, optional
        Number of noisy images to generate, by default 5.
    sigma: Union[str, float], optional
        Standard deviation for adding noise, 'auto' for automatic calculation, by default 'auto'.
    random_seed: Union[None, int], optional
        Seed for random number generator, by default None.
    return_noised_images: bool, optional
        If True, return the noisy images along with the stability value, by default False.
    device: torch.device, optional
        The device to perform calculations, by default 'cuda:0' if available, else 'cpu'.
    **kwargs
        Additional keyword arguments to be passed to the `explanator` function.

    Returns
    -------
    Union[float, Tuple[float, List[torch.Tensor]]]
        If `return_noised_images` is False, returns the stability value.
        If `return_noised_images` is True, returns a tuple containing the stability value
        and a list of noisy images.

    See Also
    --------
    stability: Measure the stability value of explanations for a set of images.

    """
    if random_seed is not None:
        rng = torch.Generator().manual_seed(random_seed)
    else:
        rng = torch.Generator()
    scaling = 0.01
    if sigma == 'auto':
        std = scaling * torch.std(image.to(float))
    elif isinstance(sigma, float) or isinstance(sigma, int):
        std = sigma
    else:
        raise Exception("Sigma must be 'auto' or numeric")

    images_to_compare = [torch.clamp(image + std * torch.randn(image.shape, generator=rng).to(device), 0, 255).round().to(image.dtype).to(device) for _ in range(n_samples)]
    epsilon = np.Inf
    stability_value = stability(explanator, transform(image), transform(torch.stack(images_to_compare)), epsilon, **kwargs)
    if return_noised_images:
        return stability_value, images_to_compare
    return stability_value


def stability(explanator: Callable[..., torch.Tensor], image: torch.Tensor,
              images_to_compare: torch.Tensor, epsilon: float = 500.0, **kwargs
              ) -> float:
    """
    Measures how similar/stable are explanations of similar photos.

    The metric measures the similarity of one type of explanation between similar photos. As explanations need to be created for each of the images close enough (by some metric, norm 2 used here) to the compared image, this may take a significant amount of processing power and memory. Values are always positive.

    The metrics is implemented as proposed in [1]_.

    Parameters
    ----------
    explanator: Callable that returns torch.Tensor
        The function used to obtain explanations for both the single image
        and the number of images in images_to_compare. Writing a wrapper to handle both options might be required.
        All **kwargs are additionaly passed to this function.
    image: torch.Tensor
        3D Tensor of the image for other images to be compared to. Shape has to be (channels, width, height).
    images_to_compare: torch.Tensor
        4D Tensor of the images compared to the original image. Shape therefore has to be (n, channels, width, height),
        where n stands for the number of images used.
    epsilon: float
        Maximal value by which an image is considered to be close enough to the original image.
        Choice of this parameter should be done carefully and
        testing by calculating some distances manually is recommended, using the matrix 2 norm.

    Returns
    -------
    float
        Value of the metrics calculated for the images close to the original image.

    See Also
    --------
    consistency: Metric representing how similar are different explanations of one photo.
    tensor_to_list_tensors: Splits first n dimensions of a Tensor into a list of Tensors.
    matrix_2_norm: Computes the 2-norm of two matrices.

    References
    ----------
    .. [1] Bobek, S., Bałaga, P., Nalepa, G.J. (2021), "Towards Model-Agnostic Ensemble Explanations."
        In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds)
        Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12745. Springer,
        Cham. https://doi.org/10.1007/978-3-030-77970-2_4

    Examples
    --------
    >>> import torch
    >>> images = torch.Tensor([0, 0.2, 0.4, 0.6, 0.8]).repeat(4, 3, 5, 1)
    >>> image = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).repeat(3, 5, 1)
    >>> def explain_dummy(images_tensor):
    ...     summed = torch.sum(images_tensor)
    ...     n = images_tensor.shape[0]
    ...     if summed > 100:
    ...         explanation = torch.Tensor([0,0,0,1,1]).repeat(n, 3, 5, 1)
    ...     else:
    ...         explanation = torch.Tensor([0,0,1,1,0]).repeat(n, 3, 5, 1)
    ...     return explanation
    >>> stability(explain_dummy, image, images, 1)
    0.13370312750339508
    """
    images_list = tensor_to_list_tensors(images_to_compare, depth=1)
    # matrix 2-norm over all 3 dimensions
    close_images = [
        other_image
        for other_image in images_list
        if matrix_2_norm(image, other_image, sum_dim=0).item() <= epsilon
    ]
    close_images_tensor = torch.stack(close_images)
    close_images_explanations = explanator(close_images_tensor, **kwargs)
    image_explanation = explanator(image.unsqueeze(dim=0), **kwargs).squeeze(dim=0)
    # matrix_2_norm works if one tensor is of one dimension bigger, casts the other to the correct size
    image_dists = matrix_2_norm(close_images_tensor, image, sum_dim=1)
    expl_dists = matrix_2_norm(close_images_explanations, image_explanation, sum_dim=1)
    return torch.max(image_dists / (expl_dists + 1)).item()


def _impact_ratio_helper(
        images_tensor: torch.Tensor,
        predictor: Callable[..., torch.Tensor],
        explanations: torch.Tensor,
        explanation_threshold: float,
        replace_value: float = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for predicting on the input and the input masked by explanations.

    This wrapper return probabilites of the model calculated for both the input,
    and the probabilites for input with significant area found by the explanation covered.
    These calculations are required in both the decision_impact_ratio and confidence_impact_ratio.

    Parameters
    ----------
    images_tensor: torch.Tensor
        The images for the prediction with shape of (n, channels, width, height), where n stands for the number of images.
    predictor: Callable that returns torch.Tensor
        Function returning a Tensor with probabilities for classification of each image to each class.
        In typical cases it's the model prediction function, possibly wrapped in a softmax function.
    explanations: torch.Tensor
        Explanations for each of the images in images_tensor. Therefore the shape should be the same as that of images_tensor.
    explanation_threshold: float
        Maximal value for explanation data point to not be considered a part of the critical area.
    replace_value: float
        The value with which data in critical area found by explanation in the image will be replaced by.

    Returns
    -------
    probabilities_original: torch.Tensor
        Probabilites of predictions calculated for the original images.
    probabilities_modified: torch.Tensor
        Probabilites of predictions calculated for the images modified by covering the critical area of explanations.

    See Also
    --------
    decision_impact_ratio: Measures the average number of changes in the predictions after hiding the critical area.
    confidence_impact_ratio: Measures the average change in probabilities after hiding the critical area.
    replace_masks: Replaces values in Tensor indexed by a boolean tensor.

    Examples
    --------
    >>> import torch
    >>> data = torch.stack([torch.ones(3, 5, 5), torch.zeros(3,5,5)])
    >>> ex_explanation = torch.BoolTensor([True, False, False, False, False]).repeat(2, 3, 5, 1)
    >>> def predictor(input_tensor):
    ...     n = input_tensor.shape[0]
    ...     if input_tensor[0,0,0,0].item() == 1:
    ...         val = torch.Tensor([0.8, 0.2, 0]).repeat(n, 1)
    ...     else:
    ...         val = torch.Tensor([0, 0.8, 0.2]).repeat(n, 1)
    ...     return val
    >>> _impact_ratio_helper(data, predictor, ex_explanation, 0.5, 0)
    (tensor([[0.8000, 0.2000, 0.0000],
         [0.8000, 0.2000, 0.0000]]),
    tensor([[0.0000, 0.8000, 0.2000],
         [0.0000, 0.8000, 0.2000]]))
    """
    probabilities_original = predictor(images_tensor)
    # one explanation per image
    explanations_boolean = explanations > explanation_threshold
    modified_images = replace_masks(images_tensor, explanations_boolean, replace_value)
    probabilities_modified = predictor(modified_images)
    return probabilities_original, probabilities_modified


def decision_impact_ratio(
        images_tensors: torch.Tensor,
        predictor: Callable[..., torch.Tensor],
        explanations: torch.Tensor,
        explanation_threshold: float,
        replace_value: float,
) -> float:
    """
    Measures the average number of changes in the predictions after hiding the critical area.

    Measures the average number of changes in the predictions after hiding the critical area found by the explanation.
    Values are in range [0,1], where 1 represents the best scenario,
    that is when for all images the prediction has changed after hiding the critical area.
    Implemented as proposed in [1]_.

    Parameters
    ----------
    images_tensors: torch.Tensor
        The images for the prediction with shape of (n, channels, width, height), where n stands for the number of images.
    predictor: Callable[..., torch.Tensor]
        Function returning a Tensor with probabilities for classification of each image to each class.
        In typical cases it's the model prediction function, possibly wrapped in a softmax function.
    explanations: torch.Tensor
        Explanations for each of the images in images_tensor. Therefore the shape should be the same as that of images_tensor.
    explanation_threshold: float
        Maximal value for explanation data point to not be considered a part of the critical area.
    replace_value: float
        The value with which data in critical area found by explanation in the image will be replaced by.

    Returns
    -------
    float
        The number of changes in the predictions after hiding the critica area found by the explanation.
        Equals to number of changed predictions/number of predictions.

    See Also
    --------
    confidence_impact_ratio: Measures the average change in probabilities after hiding the critical area.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> data = torch.stack([torch.ones(3, 5, 5), torch.zeros(3,5,5)])
    >>> ex_explanation = torch.BoolTensor([True, False, False, False, False]).repeat(2, 3, 5, 1)
    >>> def predictor(input_tensor):
    ...     n = input_tensor.shape[0]
    ...     if input_tensor[0,0,0,0].item() == 1:
    ...         val = torch.Tensor([0.8, 0.2, 0]).repeat(n, 1)
    ...     else:
    ...         val = torch.Tensor([0, 0.8, 0.2]).repeat(n, 1)
    ...     return val
    >>> decision_impact_ratio(data, predictor, ex_explanation, 0.5, 0)
    1.0
    """
    n = images_tensors.shape[0]
    # predictor returns probabilities in a tensor format
    probs_original, probs_modified = _impact_ratio_helper(
        images_tensors, predictor, explanations, explanation_threshold, replace_value
    )
    _, preds_original = torch.max(probs_original, 1)
    _, preds_modified = torch.max(probs_modified, 1)
    value = torch.sum(torch.not_equal(preds_original, preds_modified).float()) / n
    return value.item()


def confidence_impact_ratio(
        images_tensors: torch.Tensor,
        predictor: Callable[..., torch.Tensor],
        explanations: torch.Tensor,
        explanation_threshold: float,
        replace_value: float = 0,
        compare_to: str = "same_prediction"
) -> float:
    """
    Measures the average change in probabilities after hiding the critical area.

    Measures the average change in probabilities after hiding the critical area found by the explanation.
    When using default value of "same_prediction" for argument `compare_to` the probabilities taken into account are
    the ones correspoding to the same class, predicted on the original image. Then values are in range [-1,1],
    where 1 represents the best scenario, that is when for all images the prediction probability has changed
    from 1 to 0 after hiding the critical area. Meanwhile -1 represents the situation when original probability
    was close to 0 and after obscuring the critical area it change to 1.
    When using value of "new_prediction" for argument `compare_to` the probabilities taken into account are
    maximal probabilities, where the classes predicted are irrelevant. Therefore the values are in range [-1,1], where
    1 correspond to the situation when the model has probability of 1 on the original image and 0 on the modified image.
    On the other hand, the metric returns -1 when the model has close to 0 probability for all classes in original image,
    but after modifying it one class has probability of 1.
    For value "new_prediction", implemented as proposed in [1]_, while for "same_prediction" the only difference is
    usage of always same class predictions probabilities.


    Parameters
    ----------
    images_tensors: torch.Tensor
        The images for the prediction with shape of (n, channels, width, height), where n stands for the number of images.
    predictor: Callable[..., torch.Tensor]
        Function returning a Tensor with probabilities for classification of each image to each class.
        In typical cases it's the model prediction function, possibly wrapped in torch.nn.Softmax.
    explanations: torch.Tensor
        Explanations for each of the images in images_tensor. Therefore the shape should be the same as that of images_tensor.
    explanation_threshold: float
        Maximal value for explanation data point to not be considered a part of the critical area.
    replace_value: float
        The value with which data in critical area found by explanation in the image will be replaced by.
    compare_to:  str
        Argument deciding whether to compare the original maximum probability to the
        new maximum (value: "new_prediction") or to the new probability for the same index/class as the original one
        (value: "same_prediction").


    Returns
    -------
    float
        The average change in probabilities after hiding the critical area.
        Calculation is equal to average(probability_original - probability_hidden_area)

    See Also
    --------
    _impact_ratio_helper: Wrapper for predicting on the input and the input masked by explanations.
    decision_impact_ratio: Measures the average number of changes in the predictions after hiding the critical area.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> data = torch.stack([torch.ones(3, 5, 5), torch.zeros(3,5,5)])
    >>> ex_explanation = torch.BoolTensor([True, False, False, False, False]).repeat(2, 3, 5, 1)
    >>> def predictor(input_tensor):
    ...     n = input_tensor.shape[0]
    ...     if input_tensor[0,0,0,0].item() == 1:
    ...         val = torch.Tensor([0.8, 0.2, 0]).repeat(n, 1)
    ...     else:
    ...         val = torch.Tensor([0.2, 0.6, 0.2]).repeat(n, 1)
    ...     return val
    >>> confidence_impact_ratio(data, predictor, ex_explanation, 0.5, 0, "new_prediction")
    0.19999998807907104
    >>> confidence_impact_ratio(data, predictor, ex_explanation, 0.5, 0, "same_prediction")
    0.6000000238418579
    """
    if compare_to not in ["new_prediction", "same_prediction"]:
        raise Exception('Illegal value in "compare_to". Values should be one of: ["new_prediction", "same_prediction"]')
    probs_original, probs_modified = _impact_ratio_helper(
        images_tensors, predictor, explanations, explanation_threshold, replace_value
    )
    probs_max_original, predictions_max_original = torch.max(probs_original, 1)
    probs_max_modified = None
    if compare_to == "new_prediction":
        probs_max_modified, _ = torch.max(probs_modified, 1)
    elif compare_to == "same_prediction":
        probs_max_modified = probs_modified.gather(1, predictions_max_original.unsqueeze(dim=1)).squeeze(dim=1)
    value = torch.sum(probs_max_original - probs_max_modified) / images_tensors.shape[0]
    return value.item()


def accordance_recall(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Measures how much area of the mask has the explanation covered.

    Measures how much area of the mask has the explanation covered for each of the explanation, mask pairs in the data.
    Similar to the recall metric in standard classification task. Values are in range [0,1],
    where 1 represents scenario when the explanation area over the threshold covers the whole mask.
    Metric implemented as proposed in [1]_.

    Parameters
    ----------
    explanations: torch.Tensor
        Tensor of the explanations with shape as such (n, channels, width, height),
        where n represents the number of explanations and correlates masks and explanations.
    masks: torch.Tensor
        Tensor of the masks with 1 representing presence of the mask. Shape of the tensor should be (n, width, height),
        where n represents the number of masks and correlates masks and explanations or the same shape as explanations.
    threshold: float
        threshold value for the explanation to be considered a critical area.
        Values greater or equal than the threshold are considered important.

    Returns
    -------
    torch.Tensor
        Tensor with value of the metric for each of the pairs in explanations and masks.

    See Also
    --------
    accordance_precision: Measures how much area of the explanation is covered by the mask.
    F1_score: Measures the F1_score of recall and precision calculated on explanations and masks.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections," in IEEE Transactions on Artificial Intelligence,
        doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> a = torch.stack([cross_2d.repeat(3,1,1), plus_2d.repeat(3,1,1)])
    >>> b = torch.stack([plus_2d, cross_2d])
    >>> accordance_recall(a, b)
    tensor([0.2000, 0.2000])
    """
    # reshape mask to the same shape as explanation
    reshaped_mask = masks
    if masks.shape != explanations.shape:
        reshaped_mask = reshaped_mask.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    overlapping_matrix = intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    divisor = torch.sum(reshaped_mask != 0, dim=(-3, -2, -1))
    overlapping_area = torch.sum(overlapping_matrix, dim=(-3, -2, -1))
    value = overlapping_area / divisor
    return value


def accordance_precision(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> torch.Tensor:
    """
    Measures how much area of the explanation is covered by the mask.

    Measures how much area of the explanation is covered by the mask for each of the explanation,
    mask pairs in the data. Similar to the recall metric in standard classification task.
    Values are in range [0,1], where 1 represents scenario when the mask area covers
    the whole explanation area over the threshold.
    If no critical area is found in the explanation, returns 0.
    Metric implemented as proposed in [1]_.

    Parameters
    ----------
    explanations: torch.Tensor
        Tensor of the explanations with shape as such (n, channels, width, height), where n represents the number
        of explanations and correlates masks and explanations.
    masks: torch.Tensor
        Tensor of the masks with 1 representing presence of the mask. Shape of the tensor should be (n, width, height),
        where n represents the number of masks and correlates masks and explanations or the same shape as explanations.
    threshold: float
        threshold value for the explanation to be considered a critical area. Values greater or equal than
        the threshold are considered important.

    Returns
    -------
    torch.Tensor
        Tensor with value of the metric for each of the pairs in explanations and masks.

    See Also
    --------
    accordance_recall: Measures how much area of the mask has the explanation covered.
    intersection_mask: Calculates the intersection of two masks.
    F1_score: Measures the F1_score of recall and precision calculated on explanations and masks.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired pneumonia
        and COVID-19 respiratory infections," in IEEE Transactions on Artificial Intelligence,
        doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> a = torch.stack([cross_2d.repeat(3,1,1), plus_2d.repeat(3,1,1)])
    >>> b = torch.stack([plus_2d, cross_2d])
    >>> accordance_precision(a, b)
    tensor([0.2000, 0.2000])
    """
    reshaped_mask = masks
    if masks.shape != explanations.shape:
        reshaped_mask = reshaped_mask.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    overlapping_area = intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    divisor = torch.sum(explanations > threshold, dim=(-3, -2, -1))
    value = torch.sum(overlapping_area, dim=(-3, -2, -1)) / divisor
    # set nans to zero, since they indicate no value in explanation bigger than threshold
    value[value != value] = 0
    return value


def F1_score(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.0
) -> float:
    """
    Measures the F1 score of recall and precision calculated on explanations and masks.

    Measures the F1 score of recall and precision calculated on explanations and masks.
    Average of harmonic averages of ``accordance_recall`` and ``accordance_precision``.
    Values  are in range [0,1], where 1 corresponds to the scenario where
    critical areas in the explanations perfectly match the masks.
    Metric implemented as proposed in [1]_.

    Parameters
    ----------
    explanations: torch.Tensor
        Tensor of the explanations with shape as such (n, channels, width, height), where n represents the number of
        explanations and correlates masks and explanations.
    masks: torch.Tensor
        Tensor of the masks with 1 representing presence of the mask. Shape of the tensor should be (n, width, height),
        where n represents the number of masks and correlates masks and explanations or the same shape as explanations.
    threshold: float
        threshold value for the explanation to be considered a critical area.
        Values greater or equal than the threshold are considered important.

    Returns
    -------
    float
        F1 metric calculated with accordance_recall and accordance_precision of each explanation, mask pair.

    See Also
    --------
    accordance_recall: Measures how much area of the mask has the explanation covered.
    accordance_precision: Measures how much area of the explanation is covered by the mask.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> a = torch.stack([cross_2d.repeat(3,1,1), plus_2d.repeat(3,1,1)])
    >>> b = torch.stack([plus_2d, cross_2d])
    >>> F1_score(a, b)
    0.20000001788139343
    """
    acc_recall = accordance_recall(explanations, masks, threshold=threshold)
    acc_prec = accordance_precision(explanations, masks, threshold=threshold)
    values = 2 * (acc_recall * acc_prec) / (acc_recall + acc_prec)
    value = torch.sum(values) / values.shape[0]
    value[value != value] = 0
    return value.item()


def intersection_over_union(
        explanations: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5
) -> float:
    """
    Measures the average division of intersection area over the union area.

    Measures the average division of intersection area over the union area,
    where explanation values are over the threshold.
    Values  are in range [0,1], where 1 corresponds to the scenario where
    critical areas in the explanations perfectly match the masks.
    Metric implemented as proposed in [1]_.

    Parameters
    ----------
    explanations: torch.Tensor
        Tensor of the explanations with shape as such (n, channels, width, height),
        where n represents the number of explanations and correlates masks and explanations.
    masks: torch.Tensor
        Tensor of the masks with 1 representing presence of the mask. Shape of the tensor should be (n, width, height),
        where n represents the number of masks and correlates masks and explanations or the same shape as explanations.
    threshold: float
        threshold value for the explanation to be considered a critical area.
        Values greater or equal than the threshold are considered important.

    Returns
    -------
    float
        The calculated measure. Equal to average(intersection_area/union_area)

    See Also
    --------
    intersection_mask: Calculates the intersection of two masks.
    union_mask: Calculates the union of two masks.

    References
    ----------
    .. [1] L. Zou et al., "Ensemble image explainable AI (XAI) algorithm for severe community-acquired
        pneumonia and COVID-19 respiratory infections,"
        in IEEE Transactions on Artificial Intelligence, doi: 10.1109/TAI.2022.3153754.

    Examples
    --------
    >>> import torch
    >>> cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    >>> a = torch.stack([cross_2d.repeat(3,1,1), plus_2d.repeat(3,1,1)])
    >>> b = torch.stack([plus_2d, cross_2d])
    >>> intersection_over_union(a, b)
    0.3333333432674408
    """
    # one explanation per image
    reshaped_mask = masks
    if masks.shape != explanations.shape:
        reshaped_mask = reshaped_mask.unsqueeze(dim=1).repeat(1, explanations.shape[1], 1, 1)
    intersections = intersection_mask(explanations, reshaped_mask, threshold1=threshold)
    union_masks = union_mask(explanations, reshaped_mask, threshold1=threshold)
    sum_dims = tuple((-1 * i for i in range(len(explanations.shape) - 1, 0, -1)))
    intersections_areas = torch.sum(intersections, dim=sum_dims)
    union_areas = torch.sum(union_masks, dim=sum_dims)
    values = intersections_areas / union_areas
    value = torch.sum(values) / values.shape[0]
    return value.item()


def ensemble_score(
        weights: Union[List, torch.Tensor],
        metrics_scores: Union[List[torch.Tensor], torch.Tensor, List[float]],
) -> torch.Tensor:
    """
    Measure calculates the weighted sum of different metrics scores.

    Measure calculates the weighted sum of different metrics scores. Metric implemented as proposed in [1]_.

    Parameters
    ----------
    weights: Union[List, torch.Tensor]
        Weights for the corresponding metric scores.
    metrics_scores: Union[List[torch.Tensor], torch.Tensor, List[float]]
        Scores to be weighted and summed.

    Returns
    -------
    Torch.Tensor
        The weighted sum of weights times scores.

    References
    ----------
    .. [1] Bobek, S., Bałaga, P., Nalepa, G.J. (2021), "Towards Model-Agnostic Ensemble Explanations."
        In: Paszynski, M., Kranzlmüller, D., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M. (eds)
        Computational Science – ICCS 2021. ICCS 2021. Lecture Notes in Computer Science(), vol 12745. Springer,
        Cham. https://doi.org/10.1007/978-3-030-77970-2_4

    Examples
    --------
    >>> 1 * 3 + 2 * 5
    13
    >>> ensemble_score([1, 2], [3, 5])
    13
    >>> 1 * 5 + 2 * 3
    11
    >>> ensemble_score([1, 2], [5, 3])
    11
    """
    return sum(
        [
            weight * metric_score
            for (weight, metric_score) in zip(weights, metrics_scores)
        ]
    )
