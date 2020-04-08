import json
import os
import pathlib

import pytest

import numpy as np

import gouda

# import pytest


def test_ensure_dir():
    test_dir = gouda.ensure_dir("test_dir")
    test_dir2 = gouda.ensure_dir(gouda.GoudaPath("test_dir", use_absolute=False))
    assert test_dir == test_dir2.path
    assert os.path.isdir("test_dir")
    assert test_dir == "test_dir"

    test_dir_path = gouda.ensure_dir("test_dir", "check1")
    assert test_dir_path == os.path.join(test_dir, "check1")
    assert os.path.isdir(test_dir_path)

    pathlib.Path(os.path.join(test_dir_path, 'check2')).touch()
    with pytest.raises(ValueError):
        assert gouda.ensure_dir(test_dir_path, "check2")

    # Cleanup
    os.remove(os.path.join(test_dir_path, 'check2'))
    os.rmdir(test_dir_path)
    os.rmdir('test_dir')


def test_next_filename():
    assert gouda.next_filename("test.txt") == "test.txt"
    open('test.txt', 'w').close()
    assert gouda.next_filename("test.txt") == "test_2.txt"
    open("test_2.txt", 'w').close()
    assert gouda.next_filename("test.txt") == "test_3.txt"
    assert gouda.next_filename("test_2.txt") == "test_2_2.txt"
    # Cleanup
    os.remove("test.txt")
    os.remove("test_2.txt")


def test_save_load_json_dict():
    temp_data = {'a': 1, 'b': 2, 'c': 3}
    gouda.save_json(temp_data, 'test.json')
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json')
    for key in temp_data.keys():
        assert temp_data[key] == check_data[key]
    os.remove('test.json')


def test_save_load_json_list():
    temp_data = ['a', 'b', 'c']
    gouda.save_json(temp_data, 'test.json')
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json')
    for i in range(len(temp_data)):
        assert temp_data[i] == check_data[i]
    os.remove('test.json')


def test_save_load_json_nested():
    temp_data = [{'a': 1}, {'a': 2}, {'a': 3}]
    gouda.save_json(temp_data, 'test.json')
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json')
    for i in range(len(temp_data)):
        assert temp_data[i]['a'] == check_data[i]['a']
    os.remove('test.json')


def test_save_json_load_json_numpy():
    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, 'testx.json', embed_arrays=True, compressed=False)
    gouda.save_json(temp_data, 'test2.json', embed_arrays=False, compressed=False)
    gouda.save_json(temp_data, 'test3.json', embed_arrays=False, compressed=True)

    assert os.path.isfile('testx.json')
    assert not os.path.isfile('testx_array.npz')
    assert not os.path.isfile('testx_arrayzip.npz')
    assert os.path.isfile('test2.json')
    assert os.path.isfile('test2_array.npz')
    assert os.path.isfile('test3.json')
    assert os.path.isfile('test3_arrayzip.npz')

    with open('testx.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy_embed'
    with open('test2.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy'
    with open('test3.json', 'r') as f:
        data = json.load(f)
        assert data[-1] == 'numpy_zip'

    check_data = gouda.load_json('testx.json')
    check_data2 = gouda.load_json('test2.json')
    check_data3 = gouda.load_json('test3.json')
    np.testing.assert_array_equal(temp_data, check_data)
    np.testing.assert_array_equal(temp_data, check_data2)
    np.testing.assert_array_equal(temp_data, check_data3)

    os.remove('testx.json')
    os.remove('test2.json')
    os.remove('test3.json')
    os.remove('test2_array.npz')
    os.remove('test3_arrayzip.npz')


def test_save_load_json_numpy_list():
    temp_data = [np.arange(3, dtype=np.uint8), np.arange(4, 6, dtype=np.float32)]
    gouda.save_json(temp_data, 'test.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('test.json')
    assert len(check_data) == 2
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    np.testing.assert_array_equal(temp_data[1], check_data[1])

    gouda.save_json(temp_data, 'test2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('test2.json')
    assert len(check_data2) == 2
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    np.testing.assert_array_equal(temp_data[1], check_data2[1])

    os.remove('test.json')
    os.remove('test2.json')
    os.remove('test2_array.npz')


def test_save_load_json_numpy_dict():
    temp_data = {'a': np.arange(3, dtype=np.uint8)}
    gouda.save_json(temp_data, 'test.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('test.json')
    assert len(temp_data) == len(check_data)
    np.testing.assert_array_equal(temp_data['a'], check_data['a'])

    gouda.save_json(temp_data, 'test2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('test2.json')
    assert len(temp_data) == len(check_data2)
    np.testing.assert_array_equal(temp_data['a'], check_data2['a'])

    os.remove('test.json')
    os.remove('test2.json')
    os.remove('test2_array.npz')


def test_save_load_json_numpy_mixed():
    temp_data = [np.arange(3), 3]
    gouda.save_json(temp_data, 'test.json', embed_arrays=True, compressed=False)
    check_data = gouda.load_json('test.json')
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    assert check_data[1] == 3

    gouda.save_json(temp_data, 'test2.json', embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json('test2.json')
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    assert check_data2[1] == 3

    os.remove('test.json')
    os.remove('test2.json')
    os.remove('test2_array.npz')


def test_save_json_warning_error():
    temp_data = [1, 2, 3]
    with pytest.warns(UserWarning):
        gouda.save_json(temp_data, 'test.json', embed_arrays=True, compressed=True)
    os.remove('test.json')

    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, 'test.json', embed_arrays=False)
    gouda.save_json(temp_data.astype(np.float32), 'test2.json', embed_arrays=False)
    gouda.save_json(temp_data.reshape([1, 5]), 'test3.json', embed_arrays=False)
    os.remove('test_array.npz')
    os.rename('test2_array.npz', 'test_array.npz')
    with pytest.raises(ValueError):
        gouda.load_json('test.json')
    os.remove('test_array.npz')
    os.rename('test3_array.npz', 'test_array.npz')
    with pytest.raises(ValueError):
        gouda.load_json('test.json')

    os.remove('test.json')
    os.remove('test2.json')
    os.remove('test3.json')
    os.remove('test_array.npz')
