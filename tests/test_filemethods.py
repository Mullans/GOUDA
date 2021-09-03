import json
import os
import pathlib

import pytest

import numpy as np

import gouda

# import pytest


def test_ensure_dir():
    test_dir = gouda.GoudaPath("ScratchFiles/test_dir", use_absolute=False, ensure_dir=True)
    test_dir2 = gouda.ensure_dir("ScratchFiles/test_dir")
    assert str(test_dir) == str(test_dir2)
    assert os.path.isdir("ScratchFiles/test_dir")
    assert test_dir.path == "ScratchFiles/test_dir"

    test_dir_path = gouda.ensure_dir("ScratchFiles/test_dir", "check1")
    # Have to use test_dir.path. os.path.join uses the __fspath__, which is always absolute
    assert test_dir_path == os.path.join(test_dir.path, "check1")
    assert os.path.isdir(test_dir_path)

    pathlib.Path(os.path.join(test_dir_path, 'check2')).touch()
    with pytest.raises(ValueError):
        assert gouda.ensure_dir(test_dir_path, "check2")

    # Cleanup
    os.remove(os.path.join(test_dir_path, 'check2'))
    os.rmdir(test_dir_path)
    os.rmdir('ScratchFiles/test_dir')


def test_next_filename():
    assert gouda.next_filename("ScratchFiles/test.txt") == "ScratchFiles/test.txt"
    open('ScratchFiles/test.txt', 'w').close()
    assert gouda.next_filename("ScratchFiles/test.txt") == "ScratchFiles/test_2.txt"
    open("ScratchFiles/test_2.txt", 'w').close()
    assert gouda.next_filename("ScratchFiles/test.txt") == "ScratchFiles/test_3.txt"
    assert gouda.next_filename("ScratchFiles/test_2.txt") == "ScratchFiles/test_2_2.txt"
    # Cleanup
    os.remove("ScratchFiles/test.txt")
    os.remove("ScratchFiles/test_2.txt")


def test_basicname():
    assert gouda.basicname('anypath/morepath/test_item-here.jpg') == 'test_item-here'


def test_get_sorted_filenames():
    src_dir = gouda.GoudaPath('../ScratchFiles/sortedfiles')
    if not src_dir('test.txt').exists():
        gouda.ensure_dir(src_dir)
        for i in range(5):
            with open(gouda.next_filename(src_dir('test.txt')), 'w'):
                pass
        for i in range(3):
            with open(gouda.next_filename(src_dir('btest.txt')), 'w'):
                pass
    filenames = gouda.get_sorted_filenames(src_dir / '*.txt')
    assert [gouda.basicname(item) for item in filenames] == ['btest', 'btest_2', 'btest_3', 'test', 'test_2', 'test_3', 'test_4', 'test_5']
    os.remove(src_dir('test.txt'))
    for i in range(2, 6):
        os.remove(src_dir('test_{}.txt'.format(i)))
    os.remove(src_dir('btest.txt'))
    for i in range(2, 4):
        os.remove(src_dir('btest_{}.txt'.format(i)))


def test_save_load_json_dict():
    temp_data = {'a': 1, 'b': 2, 'c': 3}
    gouda.save_json(temp_data, 'ScratchFiles/test.json')
    assert os.path.isfile('ScratchFiles/test.json')
    check_data = gouda.load_json('ScratchFiles/test.json')
    for key in temp_data.keys():
        assert temp_data[key] == check_data[key]
    os.remove('ScratchFiles/test.json')


def test_save_load_json_list():
    temp_data = ['a', 'b', 'c']
    gouda.save_json(temp_data, 'ScratchFiles/test.json')
    assert os.path.isfile('ScratchFiles/test.json')
    check_data = gouda.load_json('ScratchFiles/test.json')
    for i in range(len(temp_data)):
        assert temp_data[i] == check_data[i]
    os.remove('ScratchFiles/test.json')


def test_save_load_json_nested():
    temp_data = [{'a': 1}, {'a': 2}, {'a': 3}]
    gouda.save_json(temp_data, 'ScratchFiles/test.json')
    assert os.path.isfile('ScratchFiles/test.json')
    check_data = gouda.load_json('ScratchFiles/test.json')
    for i in range(len(temp_data)):
        assert temp_data[i]['a'] == check_data[i]['a']
    os.remove('ScratchFiles/test.json')


def test_save_json_load_json_numpy():
    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, 'ScratchFiles/testx.json', embed_arrays=True, compressed=False)
    gouda.save_json(temp_data, 'ScratchFiles/test2.json', embed_arrays=False, compressed=False)
    gouda.save_json(temp_data, 'ScratchFiles/test3.json', embed_arrays=False, compressed=True)

    assert os.path.isfile('ScratchFiles/testx.json')
    assert not os.path.isfile('ScratchFiles/testx_array.npz')
    assert not os.path.isfile('ScratchFiles/testx_arrayzip.npz')
    assert os.path.isfile('ScratchFiles/test2.json')
    assert os.path.isfile('ScratchFiles/test2_array.npz')
    assert os.path.isfile('ScratchFiles/test3.json')
    assert os.path.isfile('ScratchFiles/test3_arrayzip.npz')

    with open('ScratchFiles/testx.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy_embed'
    with open('ScratchFiles/test2.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy'
    with open('ScratchFiles/test3.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy_zip'

    check_data = gouda.load_json('ScratchFiles/testx.json')
    check_data2 = gouda.load_json('ScratchFiles/test2.json')
    check_data3 = gouda.load_json('ScratchFiles/test3.json')
    np.testing.assert_array_equal(temp_data, check_data)
    np.testing.assert_array_equal(temp_data, check_data2)
    np.testing.assert_array_equal(temp_data, check_data3)

    os.remove('ScratchFiles/testx.json')
    os.remove('ScratchFiles/test2.json')
    os.remove('ScratchFiles/test3.json')
    os.remove('ScratchFiles/test2_array.npz')
    os.remove('ScratchFiles/test3_arrayzip.npz')


def test_save_load_json_set():
    test_data = {'a': set([1, 2, 3])}
    gouda.save_json(test_data, 'ScratchFiles/testset.json')
    check = gouda.load_json('ScratchFiles/testset.json')
    assert check['a'] == set([1, 2, 3])


def test_save_load_json_list_numpy():
    temp_data = np.arange(5, dtype=np.uint8)

    gouda.save_json([temp_data], 'ScratchFiles/testn1.json', embed_arrays=False, compressed=False)
    gouda.save_json([temp_data], 'ScratchFiles/testn2.json', embed_arrays=True, compressed=False)
    gouda.save_json([temp_data], 'ScratchFiles/testn3.json', embed_arrays=False, compressed=True)

    check1 = gouda.load_json('ScratchFiles/testn1.json')
    check2 = gouda.load_json('ScratchFiles/testn2.json')
    check3 = gouda.load_json('ScratchFiles/testn3.json')
    assert isinstance(check1, list)
    assert isinstance(check2, list)
    assert isinstance(check3, list)

    np.testing.assert_array_equal(temp_data, check1[0])
    np.testing.assert_array_equal(temp_data, check2[0])
    np.testing.assert_array_equal(temp_data, check3[0])
    os.remove('ScratchFiles/testn1.json')
    os.remove('ScratchFiles/testn2.json')
    os.remove('ScratchFiles/testn3.json')


def test_save_load_json_numpy_list():
    temp_data = [np.arange(3, dtype=np.uint8), np.arange(4, 6, dtype=np.float32)]
    gouda.save_json(temp_data, 'ScratchFiles/testnl.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('ScratchFiles/testnl.json')
    assert len(check_data) == 2
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    np.testing.assert_array_equal(temp_data[1], check_data[1])

    gouda.save_json(temp_data, 'ScratchFiles/testnl2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('ScratchFiles/testnl2.json')
    assert len(check_data2) == 2
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    np.testing.assert_array_equal(temp_data[1], check_data2[1])

    os.remove('ScratchFiles/testnl.json')
    os.remove('ScratchFiles/testnl2.json')
    os.remove('ScratchFiles/testnl2_array.npz')


def test_save_load_json_numpy_dict():
    temp_data = {'a': np.arange(3, dtype=np.uint8)}
    gouda.save_json(temp_data, 'ScratchFiles/testnd.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('ScratchFiles/testnd.json')
    assert len(temp_data) == len(check_data)
    np.testing.assert_array_equal(temp_data['a'], check_data['a'])

    gouda.save_json(temp_data, 'ScratchFiles/testnd2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('ScratchFiles/testnd2.json')
    assert len(temp_data) == len(check_data2)
    np.testing.assert_array_equal(temp_data['a'], check_data2['a'])

    os.remove('ScratchFiles/testnd.json')
    os.remove('ScratchFiles/testnd2.json')
    os.remove('ScratchFiles/testnd2_array.npz')


def test_save_load_json_numpy_mixed():
    temp_data = [np.arange(3), 3]
    gouda.save_json(temp_data, 'ScratchFiles/testm.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('ScratchFiles/testm.json')
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    assert check_data[1] == 3

    gouda.save_json(temp_data, 'ScratchFiles/testm2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('ScratchFiles/testm2.json')
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    assert check_data2[1] == 3

    os.remove('ScratchFiles/testm.json')
    os.remove('ScratchFiles/testm2.json')
    os.remove('ScratchFiles/testm2_array.npz')

    data = [np.int64(32), 'a', np.float32(18.32)]
    gouda.save_json(data, 'ScratchFiles/testm3.json')
    check = gouda.load_json('ScratchFiles/testm3.json')
    assert check[0] == 32
    assert np.dtype(check[0]) == 'int64'
    assert data[1] == 'a'
    assert isinstance(data[1], str)
    np.testing.assert_almost_equal(check[2], 18.32, decimal=5)
    assert np.dtype(check[2]) == 'float32'
    os.remove('ScratchFiles/testm3.json')


def test_save_json_warning_error():
    temp_data = [1, 2, 3]
    with pytest.warns(UserWarning):
        gouda.save_json(temp_data, 'ScratchFiles/testw.json', embed_arrays=True, compressed=True)
    os.remove('ScratchFiles/testw.json')

    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, 'ScratchFiles/testw.json', embed_arrays=False)
    gouda.save_json(temp_data.astype(np.float32), 'ScratchFiles/testw2.json', embed_arrays=False)
    gouda.save_json(temp_data.reshape([1, 5]), 'ScratchFiles/testw3.json', embed_arrays=False)
    os.remove('ScratchFiles/testw_array.npz')
    os.rename('ScratchFiles/testw2_array.npz', 'ScratchFiles/testw_array.npz')
    with pytest.raises(ValueError):
        gouda.load_json('ScratchFiles/testw.json')
    os.remove('ScratchFiles/testw_array.npz')
    os.rename('ScratchFiles/testw3_array.npz', 'ScratchFiles/testw_array.npz')
    with pytest.raises(ValueError):
        gouda.load_json('ScratchFiles/testw.json')

    os.remove('ScratchFiles/testw.json')
    os.remove('ScratchFiles/testw2.json')
    os.remove('ScratchFiles/testw3.json')
    os.remove('ScratchFiles/testw_array.npz')


def test_is_image():
    test_image = np.ones([20, 20])
    gouda.image.imwrite('ScratchFiles/test_image.png', test_image)
    assert gouda.is_image('ScratchFiles/test_image.png')
    assert gouda.is_image('ScratchFiles/.') is False
    with pytest.raises(FileNotFoundError):
        assert gouda.is_image('ScratchFiles/asdfhaksdfhklasjdhfakdhsfk.asdf')
    path = gouda.GoudaPath('ScratchFiles/test_image.png')
    assert gouda.is_image(path)

    os.remove('ScratchFiles/test_image.png')
    assert not os.path.exists('ScratchFiles/test_image.png')


def test_save_load_json_slice():
    data = slice(0, 10, None)
    gouda.save_json(data, 'ScratchFiles/tests.json')
    compare = gouda.load_json('ScratchFiles/tests.json')
    assert compare == data
    os.remove('ScratchFiles/tests.json')

    data = slice(0, 10, 2)
    gouda.save_json(data, 'ScratchFiles/tests.json')
    compare = gouda.load_json('ScratchFiles/tests.json')
    assert compare == data
    os.remove('ScratchFiles/tests.json')

    data = [slice(0, 10, 2)]
    gouda.save_json(data, 'ScratchFiles/tests.json')
    compare = gouda.load_json('ScratchFiles/tests.json')
    print(compare, data)
    assert compare == data
    os.remove('ScratchFiles/tests.json')

    data = [slice(0, 10, 2), {'this': slice(100, 230, None), 'that': np.array([1, 2, 3])}]
    gouda.save_json(data, 'ScratchFiles/tests.json')
    compare = gouda.load_json('ScratchFiles/tests.json')
    assert compare[0] == slice(0, 10, 2)
    assert isinstance(compare[1], dict)
    assert compare[1]['this'] == slice(100, 230)
    np.testing.assert_array_equal(compare[1]['that'], np.array([1, 2, 3]))
    os.remove('ScratchFiles/tests.json')
