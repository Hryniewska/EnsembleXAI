from unittest import TestCase
from EnsembleXAI import Metrics
import torch


class TestHelperFuncs(TestCase):
    images = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).repeat(2, 3, 1, 1)
    binary_plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    binary_cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    def test_replace_masks(self):
        masks = torch.BoolTensor([[[False, True, False]]]).repeat(2, 3, 1)
        val = 0.1
        replace_masks = Metrics.replace_masks(self.images, masks, value=val)
        self.assertTrue(torch.all(replace_masks[:, :, :, 1] == val))
        self.assertTrue(
            torch.equal(replace_masks[:, :, :, [0, 2]], self.images[:, :, :, [0, 2]])
        )

    def test_tensor_to_list_depth1(self):
        list_depth_1 = Metrics.tensor_to_list_tensors(self.images, 1)
        self.assertIsInstance(list_depth_1, list)
        self.assertTrue(len(list_depth_1) == self.images.shape[0])
        self.assertTrue(
            all(
                [list(list_element.shape) == [3, 3, 3] for list_element in list_depth_1]
            )
        )
        self.assertTrue(
            all(
                [
                    isinstance(list_element, torch.Tensor)
                    for list_element in list_depth_1
                ]
            )
        )

    def test_tensor_to_list_depth2(self):
        list_depth_2 = Metrics.tensor_to_list_tensors(self.images, 2)
        self.assertIsInstance(list_depth_2, list)
        expected_len = self.images.shape[0] * self.images.shape[1]
        self.assertTrue(len(list_depth_2) == expected_len)
        self.assertTrue(
            all([list(list_element.shape) == [3, 3] for list_element in list_depth_2])
        )
        self.assertTrue(
            all(
                [
                    isinstance(list_element, torch.Tensor)
                    for list_element in list_depth_2
                ]
            )
        )

    def test_matrix_norm_2_basic(self):
        value = Metrics.matrix_2_norm(torch.ones(3, 3), torch.zeros(3, 3))
        self.assertEqual(value, 3)

    def test_matrix_norm_2_tensor(self):
        values = Metrics.matrix_2_norm(torch.ones(3, 3, 3), torch.zeros(3, 3, 3))
        self.assertTrue(torch.all(values == 3))

    def test_matrix_norm_2_deep(self):
        value = Metrics.matrix_2_norm(
            2 * torch.ones(4, 4, 4), torch.zeros(4, 4, 4), sum_dim=0
        )
        value2 = Metrics.matrix_2_norm(
            2 * torch.ones(4, 4, 4), torch.zeros(4, 4, 4), sum_dim=-3
        )
        self.assertEqual(value, 16)
        self.assertEqual(value2, 16)

    def test_intersection(self):
        intersect = Metrics.intersection_mask(
            self.binary_plus_2d, self.binary_cross_2d
        )
        self.assertTrue(torch.any(intersect))
        self.assertFalse(torch.all(intersect))
        self.assertTrue(intersect[1, 1])
        intersect[1, 1] = False
        self.assertFalse(torch.any(intersect))
        self.assertFalse(torch.all(intersect))

    def test_intersection_threshold(self):
        plus_threshold_2d = torch.Tensor([[0, 0.2, 0], [0.5, 0.7, 0.5], [0, 0.2, 0]])
        blus_threshold_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        intersect = Metrics.intersection_mask(
            plus_threshold_2d, blus_threshold_2d, threshold1=0.3
        )
        self.assertTrue(torch.all(intersect[1]))
        self.assertFalse(torch.any(intersect[0, 2]))
        reverse_call = Metrics.intersection_mask(
            blus_threshold_2d, plus_threshold_2d, threshold2=0.3
        )
        self.assertTrue(torch.all(intersect == reverse_call))

    def test_union(self):
        union = Metrics.union_mask(self.binary_plus_2d, self.binary_cross_2d)
        self.assertTrue(torch.all(union))

    def test_union_threshold(self):
        union_threshold_a = torch.Tensor([[1, 0.4, 1], [1, 0.4, 1], [0, 0, 0]])
        union_threshold_b = torch.Tensor([[0, 0, 0], [0, 0, 0], [0.6, 0.6, 0.6]])
        union_all = Metrics.union_mask(union_threshold_a, union_threshold_b)
        self.assertTrue(torch.all(union_all))

        union_05 = Metrics.union_mask(
            union_threshold_a, union_threshold_b, threshold1=0.5
        )
        correct_union_05 = torch.BoolTensor(
            [[True, False, True], [True, False, True], [True, True, True]]
        )
        self.assertTrue(torch.all(union_05 == correct_union_05))

        union_07 = Metrics.union_mask(
            union_threshold_a, union_threshold_b, threshold1=0.5, threshold2=0.7
        )
        correct_union_07 = torch.BoolTensor(
            [[True, False, True], [True, False, True], [False, False, False]]
        )
        self.assertTrue(torch.all(union_07 == correct_union_07))


def _predict_dummy(tensor: torch.Tensor, classes=1000):
    return torch.ones([tensor.shape[0], classes]) / classes


def _predict_dummy2(input_tensor, n_classes=1000):
    n = input_tensor.shape[0]
    half_classes = n_classes // 2
    third_classes = n_classes // 3
    quarter_classes = n_classes // 4
    if torch.sum(input_tensor).item() > 300000:
        preds = torch.cat([torch.zeros([n, n_classes - quarter_classes]), 0.004 * torch.ones([n, quarter_classes])],
                          dim=1)
    elif torch.sum(input_tensor).item() > 200000:
        preds = torch.cat([torch.zeros([n, n_classes - third_classes]), 0.0033 * torch.ones([n, third_classes])], dim=1)
    else:
        preds = torch.cat([torch.zeros([n, n_classes - half_classes]), 0.002 * torch.ones([n, half_classes])], dim=1)
    return preds


def _explain_dummy(images_tensor):
    summed = torch.sum(images_tensor)
    n = images_tensor.shape[0]
    if summed > 100:
        explanation = torch.Tensor([0, 0, 0, 1, 1]).repeat(n, 3, 5, 1)
    else:
        explanation = torch.Tensor([0, 0, 1, 1, 0]).repeat(n, 3, 5, 1)
    return explanation


class TestMetrics(TestCase):
    binary_plus_2d = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    binary_cross_2d = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    plus_mask = binary_plus_2d.unsqueeze(0)
    cross_mask = binary_cross_2d.unsqueeze(0)
    plus_expl = binary_plus_2d.unsqueeze(0).unsqueeze(0)
    cross_expl = binary_cross_2d.unsqueeze(0).unsqueeze(0)
    half_plus_expl = plus_expl / 2

    def test_consistency(self):
        ones = torch.Tensor([1]).repeat(4, 3, 10, 10)
        consist = Metrics.consistency(ones)
        self.assertTrue(consist == 1)
        twos = 2 * torch.Tensor([1]).repeat(4, 3, 10, 10)
        oneandtwo = torch.cat([ones, twos], dim=0)
        consist2 = Metrics.consistency(oneandtwo)
        self.assertAlmostEqual(consist2, 0.05458, 4)

    def test_stability(self):
        images = torch.Tensor([0, 0.2, 0.4, 0.6, 0.8]).repeat(4, 3, 5, 1)
        image = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).repeat(3, 5, 1)
        value = Metrics.stability(_explain_dummy, image, images, 1)
        self.assertAlmostEqual(value, 0.1337, 5)

    def test_impact_ratio_helper_simple(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.randn([n, 3, 32, 32])
        baseline = 0
        threshold = 0.5
        org, mod = Metrics._impact_ratio_helper(
            images_tensor, _predict_dummy, expls, threshold, baseline
        )
        self.assertIsInstance(org, torch.Tensor)
        self.assertIsInstance(mod, torch.Tensor)
        self.assertTrue(torch.all(org == 0.0010))
        self.assertTrue(torch.all(org == mod))

    def test_impact_ratio_helper_complex(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.Tensor([0, 0.3, 0.6, 1]).repeat(8).repeat(n, 3, 32, 1)
        org, mod = Metrics._impact_ratio_helper(
            images_tensor, _predict_dummy2, expls, 0.4, 0
        )
        self.assertIsInstance(org, torch.Tensor)
        self.assertIsInstance(mod, torch.Tensor)
        self.assertTrue(torch.all(org[:, 750:] == 0.004))
        self.assertTrue(torch.all(org[:, :750] == 0.0))
        self.assertFalse(torch.all(mod == org))

    def test_decision_impact_ratio_simple(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.randn([n, 3, 32, 32])
        baseline = 0
        threshold = 0.5
        value = Metrics.decision_impact_ratio(
            images_tensor, _predict_dummy, expls, threshold, baseline
        )
        self.assertEqual(value, 0)

    def test_decision_impact_ratio_complex(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.Tensor([0, 0.3, 0.6, 1]).repeat(8).repeat(n, 3, 32, 1)
        val0 = Metrics.decision_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 0)
        val1 = Metrics.decision_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 1)
        self.assertEqual(val0, 1)
        self.assertEqual(val1, 0)

    def test_confidence_impact_ratio_simple(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.randn([n, 3, 32, 32])
        baseline = 0
        threshold = 0.5
        value1 = Metrics.confidence_impact_ratio(
            images_tensor, _predict_dummy, expls, threshold, baseline, "new_prediction"
        )
        value2 = Metrics.confidence_impact_ratio(
            images_tensor, _predict_dummy, expls, threshold, baseline, "same_prediction"
        )
        self.assertEqual(value1, 0)
        self.assertEqual(value2, 0)

    def test_confidence_impact_ratio_complex(self):
        n = 100
        images_tensor = torch.ones([n, 3, 32, 32])
        expls = torch.Tensor([0, 0.3, 0.6, 1]).repeat(8).repeat(n, 3, 32, 1)
        val1 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 0.4, "new_prediction")
        val2 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 0.7, "new_prediction")
        val3 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 1, "new_prediction")
        self.assertAlmostEqual(val1, 0.002, 5)
        self.assertAlmostEqual(val2, 0.0007, 5)
        self.assertAlmostEqual(val3, 0, 5)
        val4 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 0.4, "same_prediction")
        val5 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 0.7, "same_prediction")
        val6 = Metrics.confidence_impact_ratio(images_tensor, _predict_dummy2, expls, 0, 1, "same_prediction")
        self.assertAlmostEqual(val4, 0.002, 5)
        self.assertAlmostEqual(val5, 0.0007, 5)
        self.assertAlmostEqual(val6, 0, 5)

    def test_accordance_recall(self):
        val1 = Metrics.accordance_recall(self.plus_expl, self.plus_mask).item()
        self.assertAlmostEqual(val1, 1, 4)
        val2 = Metrics.accordance_recall(self.plus_expl, self.cross_mask).item()
        self.assertAlmostEqual(val2, 0.2, 4)
        val3 = Metrics.accordance_recall(self.cross_expl, self.plus_mask).item()
        self.assertAlmostEqual(val3, 0.2, 4)

    def test_accordance_recall_threshold(self):
        val = Metrics.accordance_recall(
            self.half_plus_expl, self.plus_mask, threshold=0.6
        ).item()
        self.assertAlmostEqual(val, 0, 4)

    def test_accordance_precision(self):
        val1 = Metrics.accordance_precision(self.plus_expl, self.plus_mask).item()
        self.assertAlmostEqual(val1, 1, 4)
        val2 = Metrics.accordance_precision(self.plus_expl, self.cross_mask).item()
        self.assertAlmostEqual(val2, 0.2, 4)
        val3 = Metrics.accordance_precision(self.cross_expl, self.plus_mask).item()
        self.assertAlmostEqual(val3, 0.2, 4)

    def test_accordance_precision_threshold(self):
        val = Metrics.accordance_precision(
            1.5 * self.half_plus_expl, self.plus_mask, threshold=0.5
        ).item()
        self.assertAlmostEqual(val, 1, 4)

    def test_f1_score(self):
        explanations = torch.cat((self.plus_expl, self.cross_expl))
        masks = torch.cat((self.cross_mask, self.plus_mask))
        val = Metrics.F1_score(explanations, masks)
        self.assertAlmostEqual(val, 0.2, 4)

    def test_iou(self):
        explanations = torch.cat((self.plus_expl, self.cross_expl))
        masks = torch.cat((self.cross_mask, self.plus_mask))
        val = Metrics.intersection_over_union(explanations, masks)
        self.assertAlmostEqual(val, 0.11111, 4)

    def test_ensemble_score(self):
        # Metrics.ensemble_score()
        self.assertEqual(Metrics.ensemble_score([1, 2], [3, 5]), 13)
        self.assertEqual(Metrics.ensemble_score([1, 2], [5, 3]), 11)
