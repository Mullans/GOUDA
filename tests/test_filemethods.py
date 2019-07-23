import os

import numpy as np

import gouda

# import pytest


def test_ensure_dir():
    test_dir = gouda.ensure_dir("test_dir")
    test_dir2 = gouda.ensure_dir("test_dir")
    assert test_dir == test_dir2
    assert os.path.isdir("test_dir")
    assert test_dir == "test_dir"
    # Cleanup
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
    gouda.save_json(temp_data, 'test.json', numpy=True)
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json', numpy=True)
    np.testing.assert_array_equal(temp_data, check_data)
    check_data2 = gouda.load_json('test.json', numpy=False)
    assert check_data2['dtype'] == 'uint8'
    assert isinstance(check_data2['numpy_array'], list)
    assert len(check_data2['numpy_array']) == 5
    assert check_data2['shape'][0] == 5
    os.remove('test.json')


def test_save_load_json_numpy_list():
    temp_data = [np.arange(3, dtype=np.uint8), np.arange(4, 6, dtype=np.float32)]
    gouda.save_json(temp_data, 'test.json', numpy=True)
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json', numpy=True)
    assert len(check_data) == 2
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    np.testing.assert_array_equal(temp_data[1], check_data[1])
    check_data2 = gouda.load_json('test.json', numpy=False)
    assert check_data2[0]['dtype'] == 'uint8'
    assert check_data2[1]['dtype'] == 'float32'
    assert check_data2[0]['numpy_array'][0] == 0
    assert check_data2[0]['numpy_array'][-1] == 2
    assert check_data2[1]['numpy_array'][0] == 4
    assert check_data2[1]['numpy_array'][-1] == 5
    assert check_data2[0]['shape'][0] == 3
    assert check_data2[1]['shape'][0] == 2
    os.remove('test.json')


def test_save_load_json_numpy_dict():
    temp_data = {'a': np.arange(3, dtype=np.uint8)}
    gouda.save_json(temp_data, 'test.json', numpy=True)
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json', numpy=True)
    np.testing.assert_array_equal(temp_data['a'], check_data['a'])
    check_data2 = gouda.load_json('test.json', numpy=False)
    assert check_data2['a']['dtype'] == 'uint8'
    assert check_data2['a']['numpy_array'][0] == 0
    assert check_data2['a']['numpy_array'][-1] == 2
    assert check_data2['a']['shape'][0] == 3
    os.remove('test.json')


def test_save_load_json_numpy_mixed():
    temp_data = [np.arange(3), 3]
    gouda.save_json(temp_data, 'test.json', numpy=True)
    assert os.path.isfile('test.json')
    check_data = gouda.load_json('test.json', numpy=True)
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    assert check_data[1] == 3
    os.remove('test.json')
