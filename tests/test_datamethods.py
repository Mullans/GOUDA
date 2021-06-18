import pytest

import numpy as np

import gouda


def test_num_digits():
    assert gouda.num_digits(5) == 1
    assert gouda.num_digits(9) == 1
    assert gouda.num_digits(10) == 2
    assert gouda.num_digits(99) == 2
    assert gouda.num_digits(100) == 3
    assert gouda.num_digits(0) == 1
    assert gouda.num_digits(-1) == 1
    assert gouda.num_digits(-10) == 2


def test_arr_sample():
    temp_data = np.arange(1, 5)
    resampled = gouda.arr_sample(temp_data, 2)
    np.testing.assert_array_equal(resampled, np.array([1, 3]))
    resampled2 = gouda.arr_sample(temp_data, 0.5)
    np.testing.assert_array_equal(resampled2, np.array([1, 1, 2, 2, 3, 3, 4, 4]))


def test_arr_sample_exception():
    temp_data = np.arange(1, 5).reshape([2, 2])
    with pytest.raises(ValueError):
        assert gouda.arr_sample(temp_data, 1)


def test_sigmoid():
    np.testing.assert_almost_equal(gouda.sigmoid(0), 0.5)
    np.testing.assert_almost_equal(gouda.sigmoid(-100), 0)
    np.testing.assert_almost_equal(gouda.sigmoid(100), 1)


def test_normalize():
    test_data = np.arange(1000).reshape([10, 10, 10])
    normed_1 = gouda.normalize(test_data)
    np.testing.assert_almost_equal(normed_1.std(), 1)
    assert normed_1.mean() == 0

    normed_alt = gouda.normalize(test_data.astype(np.float))
    np.testing.assert_equal(normed_1, normed_alt)

    normed_2 = gouda.normalize(test_data, axis=1)
    np.testing.assert_array_almost_equal(normed_2.mean(axis=1), np.zeros([10, 10]))
    np.testing.assert_equal(normed_2.std(axis=1), np.ones([10, 10]))

    normed_3 = gouda.normalize(test_data, axis=(0, 1))
    np.testing.assert_array_almost_equal(normed_3.mean(axis=(0, 1)), np.zeros([10, ]))
    np.testing.assert_equal(normed_3.std(axis=(0, 1)), 1)


def test_rescale():
    test_data = np.arange(100).reshape([10, 10])
    scaled_1 = gouda.rescale(test_data, new_min=0, new_max=1, axis=1)
    manual = (test_data - test_data.min(axis=1, keepdims=True)) / (test_data.max(axis=1, keepdims=True) - test_data.min(axis=1, keepdims=True))
    np.testing.assert_array_equal(scaled_1, manual)
    scaled_alt = gouda.rescale(test_data.astype(np.float), new_min=0, new_max=1, axis=1)
    np.testing.assert_array_equal(scaled_1, scaled_alt)

    scaled_2 = gouda.rescale(test_data, new_min=-1, new_max=2)
    assert scaled_2.max() == 2
    assert scaled_2.min() == -1


def test_factors():
    result1 = gouda.factors(100)
    expected1 = set([1, 2, 4, 5, 10, 20, 25, 50, 100])
    assert len(result1.symmetric_difference(expected1)) == 0

    result2 = gouda.factors(6)
    expected2 = set([1, 2, 3, 6])
    assert len(result2.symmetric_difference(expected2)) == 0

    result3 = gouda.factors(7)
    expected3 = set([1, 7])
    assert len(result3.symmetric_difference(expected3)) == 0

    with pytest.raises(ValueError):
        assert gouda.factors(0)

    with pytest.warns(UserWarning):
        result4 = gouda.factors(-1)
        expected4 = set([1])
        assert len(result4.symmetric_difference(expected4)) == 0


def test_prime_factors():
    result1 = gouda.prime_factors(100)
    expected1 = [2, 2, 5, 5]
    assert result1 == expected1

    result2 = gouda.prime_factors(7)
    expected2 = [7]
    assert result2 == expected2

    with pytest.raises(ValueError):
        assert gouda.prime_factors(0)

    with pytest.warns(UserWarning):
        result3 = gouda.prime_factors(-1)
        expected3 = [1]
        assert result3 == expected3


def test_prime_overlap():
    result1 = gouda.prime_overlap(2, 5)
    assert len(result1) == 0

    result2 = gouda.prime_overlap(4, 10)
    assert result2 == [2]

    result3 = gouda.prime_overlap(672, 42)
    assert result3 == [2, 3, 7]

    result4 = gouda.prime_overlap(42, 672)
    assert result4 == [2, 3, 7]

    result5 = gouda.prime_overlap(7, 5)
    assert result5 == []


def test_flip_dict():
    one2one_dict = {'a': 1, 'b': 2, 'c': 3}
    many2one_dict = {'a': 1, 'b': 1, 'c': 2}
    all2one_dict = {'a': 1, 'b': 1, 'c': 1}

    flip1 = gouda.flip_dict(one2one_dict)
    assert flip1 == {1: 'a', 2: 'b', 3: 'c'}

    flip2 = gouda.flip_dict(one2one_dict, unique_items=True)
    assert flip2 == {1: 'a', 2: 'b', 3: 'c'}

    flip3 = gouda.flip_dict(one2one_dict, force_list_values=True)
    assert flip3 == {1: ['a'], 2: ['b'], 3: ['c']}

    flip4 = gouda.flip_dict(many2one_dict)
    assert flip4 == {1: ['a', 'b'], 2: 'c'}

    flip5 = gouda.flip_dict(many2one_dict, unique_items=True)
    assert flip5 == {1: 'b', 2: 'c'}

    flip6 = gouda.flip_dict(many2one_dict, force_list_values=True)
    assert flip6 == {1: ['a', 'b'], 2: ['c']}

    flip7 = gouda.flip_dict(all2one_dict)
    assert flip7 == {1: ['a', 'b', 'c']}


def test_softmax():
    data = np.arange(10).reshape([5, 2])
    assert gouda.softmax(data).sum() == 1
    assert gouda.softmax(data, axis=0).sum() == 2
    assert gouda.softmax(data, axis=1).sum() == 5

    np.testing.assert_array_equal(gouda.softmax(data), gouda.softmax(data.astype(np.float)))


def test_roc_mcc():
    label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fps, tps, thresh = gouda.roc_curve(label, pred, as_rates=False)
    np.testing.assert_array_equal(np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5]), fps)
    np.testing.assert_array_equal(np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]), tps)
    np.testing.assert_array_equal(np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]), thresh)

    label_alt = np.array(label)
    pred_alt = np.array(pred)
    fps_alt, tps_alt, thresh_alt = gouda.roc_curve(label_alt, pred_alt, as_rates=False)
    np.testing.assert_array_equal(fps, fps_alt)
    np.testing.assert_array_equal(tps, tps_alt)
    np.testing.assert_array_equal(thresh, thresh_alt)

    fpr, tpr, thresh2 = gouda.roc_curve(label, pred, as_rates=True)
    np.testing.assert_array_equal(fps / 5, fpr)
    np.testing.assert_array_equal(tps / 5, tpr)
    np.testing.assert_array_equal(thresh, thresh2)

    opt_mcc, opt_thresh = gouda.optimal_mcc_from_roc(fps, tps, thresh, optimal_only=True)
    assert opt_mcc == 1
    assert opt_thresh == 0.5

    all_mcc, mcc_thresh = gouda.optimal_mcc_from_roc(fps, tps, thresh, optimal_only=False)
    check_arr = np.array([0.0000, 0.3333, 0.5000, 0.6547, 0.8165, 1.0000, 0.8165, 0.6547, 0.5000, 0.3333, 0.0000])
    np.testing.assert_array_almost_equal(all_mcc, check_arr, decimal=4)
    np.testing.assert_array_equal(mcc_thresh, thresh)

    curve, mcc_thresh2 = gouda.mcc_curve(label, pred)
    np.testing.assert_array_equal(all_mcc, curve)
    np.testing.assert_array_equal(mcc_thresh, mcc_thresh2)


def test_spec_at_sens():
    label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2]
    fps, tps, thresh = gouda.roc_curve(label, pred, as_rates=False)

    low_spec, med_spec, high_spec = gouda.spec_at_sens(label, pred, sensitivities=[0, 0.9, 1])
    assert low_spec == 1.0
    assert med_spec == 0.4
    assert high_spec == 0.4

    [med_alt] = gouda.spec_at_sens(np.array(label), np.array(pred), sensitivities=0.9)
    assert med_spec == med_alt


def test_get_confusion_stats_dice_jacaard():
    label = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2]
    true_pos, false_pos, true_neg, false_neg = gouda.get_confusion_stats(label, pred, threshold=0.5)
    assert true_pos == 4
    assert false_pos == 0
    assert true_neg == 4
    assert false_neg == 1

    true_pos, false_pos, true_neg, false_neg = gouda.get_confusion_stats(np.array(label), np.array(pred), threshold=0.2)
    assert true_pos == 5
    assert false_pos == 3
    assert true_neg == 1
    assert false_neg == 0

    assert gouda.dice_coef(label, pred, 0.5) == 8 / (8 + 0 + 1)
    assert gouda.jaccard_coef(label, pred, 0.5) == 4 / (4 + 1 + 0)


def test_value_crossing():
    data = [-1, -1, 0, 1, 0, -1, -1, 0, 0, -1, 0, 1, 0, 1]
    counts = gouda.value_crossing(np.array(data))
    idx = gouda.value_crossing(data, return_indices=True)
    assert counts == idx.size
    pos_only = gouda.value_crossing(data, negative_crossing=False)
    neg_only = gouda.value_crossing(data, positive_crossing=False)
    assert pos_only == 2
    assert neg_only == 1
    assert pos_only + neg_only == counts

    with pytest.raises(ValueError):
        gouda.value_crossing(data, negative_crossing=False, positive_crossing=False)


def test_center_of_mass():
    data = np.zeros([5, 5])
    with pytest.raises(ValueError):
        gouda.center_of_mass(data)

    data = np.ones([5, 5])
    np.testing.assert_array_equal(gouda.center_of_mass(data), np.array([2, 2]))

    data[:2] = 0
    data[:, :2] = 0
    np.testing.assert_array_equal(gouda.center_of_mass(data), np.array([3, 3]))

    data = np.ones([5, 5, 5])
    np.testing.assert_array_equal(gouda.center_of_mass(data), np.array([2, 2, 2]))
    data[:2] = 0
    np.testing.assert_array_equal(gouda.center_of_mass(data), np.array([3, 2, 2]))


def test_accuracy_curve():
    label = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    acc, thresh = gouda.accuracy_curve(label, pred)
    np.testing.assert_array_equal(thresh, np.array(pred)[::-1])
    counts = np.array([6, 7, 8, 9, 10, 10, 9, 8, 7, 6])
    np.testing.assert_array_equal(acc, counts / 10)

    acc2, thresh2, peak_acc, peak_thresh = gouda.accuracy_curve(label, pred, return_peak=True)
    np.testing.assert_array_equal(acc, acc2)
    np.testing.assert_array_equal(thresh, thresh2)
    assert peak_acc in acc
    assert peak_acc == 1
    assert peak_thresh == 0.5

    label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
    acc, thresh = gouda.accuracy_curve(label, pred)
    np.testing.assert_array_equal(thresh, pred)
    counts = np.array([5, 4, 3, 2, 1, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(acc, counts / 10)


def test_clip():
    x = np.arange(300)
    result = gouda.clip(x, output_min=0, output_max=1, input_min=0, input_max=255)
    assert result.min() == 0
    assert result.max() == 1
    assert (result == 1).sum() == 45
    assert (result == 0).sum() == 1
    np.testing.assert_array_almost_equal(result[:256], np.linspace(0, 1, num=256))

    result = gouda.clip(x, output_min=0, output_max=1, input_min=0, input_max=300)
    np.testing.assert_array_almost_equal(result, x / 300.0, decimal=8)

    result = gouda.clip(x, output_min=1, output_max=2, input_min=0, input_max=299)
    np.testing.assert_array_almost_equal(result, (x / 299) + 1)
    assert result.min() == 1

    result = gouda.clip(x, output_min=1, output_max=2, input_min=0, input_max=500)
    np.testing.assert_almost_equal(result, x / 500 + 1)
    assert result.min() == 1
    np.testing.assert_almost_equal(result.max(), 299 / 500 + 1)
