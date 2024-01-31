import torch as t

from EnsembleXAI import Ensemble
from unittest import TestCase

from EnsembleXAI.Ensemble import _normalize_across_dataset, _reformat_input_tensors


def _dummy_metric(x: t.Tensor) -> t.Tensor:
    return t.ones([x.size(dim=0)])


def _dummy_metric2(x: t.Tensor) -> t.Tensor:
    return 2 * _dummy_metric(x)


class TestReformatInputs(TestCase):
    def test_tensors(self):
        actual = _reformat_input_tensors(t.zeros([1, 3, 32, 32]))
        expected = t.zeros((1, 1, 3, 32, 32))
        self.assertTrue(t.equal(actual, expected))

        actual = _reformat_input_tensors(t.zeros([3, 1, 32, 32]))
        expected = _reformat_input_tensors(t.zeros([1, 3, 1, 32, 32]))
        self.assertTrue(t.equal(actual, expected))

    def test_tuples(self):

        x = t.tensor([[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                       [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]], dtype=t.float)
        y = x

        actual = _reformat_input_tensors((x, y))
        expected = t.tensor([[[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                              [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]],
                             [[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                              [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]]
                             ], dtype=t.float)
        self.assertTrue(t.equal(actual, expected))


class TestSupervisedXAI(TestCase):
    def test_ensemble_mult_channels_mult_obs(self):
        inputs = t.rand([90, 3, 3, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        ensembled = Ensemble.supervisedXAI(inputs, masks, shuffle=False)
        self.assertTrue(ensembled.shape == (90, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_mult_obs(self):
        inputs = t.rand([90, 3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        ensembled = Ensemble.supervisedXAI(inputs, masks, shuffle=False)
        self.assertTrue(ensembled.shape == (90, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_mult_obs_weights_auto(self):
        inputs = t.rand([90, 3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        ensembled = Ensemble.supervisedXAI(inputs, masks, shuffle=False, weights='auto')
        self.assertTrue(ensembled.shape == (90, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_mult_obs_weights_tensor(self):
        inputs = t.rand([90, 3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        weights = t.rand(90)
        ensembled = Ensemble.supervisedXAI(inputs, masks, shuffle=False, weights=weights)
        self.assertTrue(ensembled.shape == (90, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape

    def test_ensemble_one_channel_mult_obs_weights_numpy(self):
        inputs = t.rand([90, 3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[90, 32, 32])
        weights = t.rand(90).numpy()
        ensembled = Ensemble.supervisedXAI(inputs, masks, shuffle=False, weights=weights)
        self.assertTrue(ensembled.shape == (90, 32, 32))
        # hard to predict outcome of this algorithm to check exact correctness, even on not random data
        # for now testing only result's shape
    def test_ensemble_one_channel_one_obs(self):
        inputs = t.rand([3, 1, 32, 32])
        masks = t.randint(low=0, high=2, size=[1, 32, 32])
        with self.assertRaises(AssertionError):
            Ensemble.supervisedXAI(inputs, masks, shuffle=False)
            Ensemble.supervisedXAI(inputs, masks, shuffle=False, n_folds=1)


class TestNormalize(TestCase):
    def test_normalization(self):
        x = t.tensor([[[[[1, 2], [2, 1]]], [[[3, 4], [3, 4]]]],
                      [[[[3, 5], [5, 3]]], [[[0, 1], [1, 0]]]]], dtype=t.float64)
        normalized = _normalize_across_dataset(x)
        expected = t.tensor([[[[[-1.1068, -0.4743],
                                [-0.4743, -1.1068]]],

                              [[[0.5916, 1.1832],
                                [0.5916, 1.1832]]]],

                             [[[[0.1581, 1.4230],
                                [1.4230, 0.1581]]],

                              [[[-1.1832, -0.5916],
                                [-0.5916, -1.1832]]]]], dtype=t.float64)
        self.assertTrue(t.allclose(normalized, expected, atol=0.01))


    def test_normalization_sample(self):
        x = t.rand(8, 3, 3, 512, 512)
        normalized = _normalize_across_dataset(x)
        self.assertEqual(normalized.shape, x.shape)


class TestAutoweighted(TestCase):
    x = t.tensor([[[[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                   [[[0, 1], [1, 0]], [[0, 1], [1, 0]]]]], dtype=t.float)
    y = t.squeeze(x, 0)

    def test_ensemble_single_obs_single_metric_mult_channel(self):
        ensemble = Ensemble.autoweighted(self.x, [1], [_dummy_metric])
        self.assertIsInstance(ensemble, t.Tensor)

        expected = t.tensor([[[[-0.9354, 0.9354],
                              [0.9354, -0.9354]],
                             [[-0.9354, 0.9354],
                              [0.9354, -0.9354]]]])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))


    def test_ensemble_single_obs_precomputed_metric_mult_channel(self):
        ensemble = Ensemble.autoweighted(self.x, [1], precomputed_metrics=t.ones((1, 2, 1)))
        self.assertIsInstance(ensemble, t.Tensor)

        expected = t.tensor([[[[-0.9354, 0.9354],
                              [0.9354, -0.9354]],
                             [[-0.9354, 0.9354],
                              [0.9354, -0.9354]]]])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_mult_obs_mult_metric_single_channel(self):
        ensemble = Ensemble.autoweighted(self.x, [0.5, 0.5], [_dummy_metric, _dummy_metric])
        self.assertIsInstance(ensemble, t.Tensor)

        expected = t.tensor([[[-0.9354, 0.9354],
                              [0.9354, -0.9354]],
                             [[-0.9354, 0.9354],
                              [0.9354, -0.9354]]])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_one_obs_one_channel_one_metric(self):
        exp1 = t.tensor([[[[0, 1], [1, 0]]], [[[0, 1], [1, 0]]]], dtype=t.float)
        ensemble = Ensemble.autoweighted(tuple(exp1), [1], [_dummy_metric])
        expected = t.tensor([[[[[-0.8660, 0.8660],
                               [0.8660, -0.8660]]]]])
        self.assertTrue(t.allclose(ensemble, expected, atol=.01))

    def test_ensemble_multiple_obs_multiple_channel_single_metric(self):
        ensemble = Ensemble.autoweighted((self.y, self.y), [1], [_dummy_metric])
        expected = t.tensor([[[[-0.9682, 0.9682],
                               [0.9682, -0.9682]],
                              [[-0.9682, 0.9682],
                               [0.9682, -0.9682]]],
                             [[[-0.9682, 0.9682],
                               [0.9682, -0.9682]],
                              [[-0.9682, 0.9682],
                               [0.9682, -0.9682]]]
                             ])

        self.assertTrue(t.allclose(ensemble, expected, atol=.01))


class TestNormEnsembleXAI(TestCase):
    exp1 = t.ones([1, 2, 2])
    exp3 = 3 * t.ones([1, 2, 2])
    obs1_tensor = t.stack((exp1, exp3))
    exp0 = t.zeros([1, 2, 2])
    obs2_tensor = t.stack([exp0, exp3])

    obs3_tensor = t.stack((exp1, exp0, exp3)).squeeze()
    mult_obs_tensor = t.stack([obs1_tensor, obs2_tensor])

    def test_one_obs_mult_channels(self):
        ensembled = Ensemble.normEnsembleXAI(self.obs3_tensor, 'avg')
        self.assertIsInstance(ensembled, t.Tensor)

        self.assertTrue(t.equal(ensembled, self.obs3_tensor[None, :]))

    def test_one_obs_one_channel_avg(self):
        # tuple input
        ensembled = Ensemble.normEnsembleXAI((self.exp1, self.exp3), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 2 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_one_channel_max(self):
        ensembled = Ensemble.normEnsembleXAI((self.exp1, self.exp3), 'max')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = 3 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, 'max')
        self.assertTrue(t.equal(ensembled, expected))

    def test_one_obs_min(self):
        ensembled = Ensemble.normEnsembleXAI((self.exp1, self.exp3), 'min')
        self.assertIsInstance(ensembled, t.Tensor)

        expected = t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
        # tensor input
        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, 'min')
        self.assertTrue(t.equal(ensembled, expected))

    def test_multi_obs_avg(self):
        # tuple input
        ensembled = Ensemble.normEnsembleXAI((self.obs1_tensor, self.obs2_tensor), 'avg')
        self.assertIsInstance(ensembled, t.Tensor)
        expected = t.stack([2 * t.ones([1, 2, 2]), 1.5 * t.ones([1, 2, 2])])
        self.assertTrue(t.equal(ensembled, expected))

        # tensor input
        ensembled = Ensemble.normEnsembleXAI(self.mult_obs_tensor, 'avg')
        self.assertTrue(t.equal(ensembled, expected))

    def test_max_abs_aggregation(self):
        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, 'max_abs')
        self.assertIsInstance(ensembled, t.Tensor)
        expected = self.exp3.unsqueeze(0)
        self.assertTrue(t.equal(ensembled, expected))

    def test_illegal_args(self):
        with self.assertRaises(AssertionError):
            Ensemble.normEnsembleXAI(self.obs1_tensor, 'asdf')
            Ensemble.normEnsembleXAI(self.obs1_tensor, 2)

    def test_custom_func(self):
        def custom_avg(x):
            return sum(x) / len(x)

        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, custom_avg)

        self.assertIsInstance(ensembled, t.Tensor)

        expected = 2 * t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))

        # 2 observations
        ensembled = Ensemble.normEnsembleXAI((self.obs1_tensor, self.obs2_tensor), custom_avg)
        self.assertIsInstance(ensembled, t.Tensor)

        expected = t.stack([2 * t.ones([1, 2, 2]), 1.5 * t.ones([1, 2, 2])])
        self.assertTrue(t.equal(ensembled, expected))

        def custom_func(x):
            return (3 * x[0] + x[1]) / 6

        ensembled = Ensemble.normEnsembleXAI(self.obs1_tensor, custom_func)
        expected = t.ones([1, 1, 2, 2])
        self.assertTrue(t.equal(ensembled, expected))
