from __future__ import annotations

import pytest

import json
import numpy as np
import os
import pathlib
import re

import gouda
import gouda.image


def test_ensure_dir(scratch_path):
    test_dir_path = gouda.ensure_dir(scratch_path / "test_dir", "check1")
    assert not os.path.exists(os.path.join(test_dir_path, "check2"))
    gouda.ensure_dir(test_dir_path)

    test_dir = gouda.GoudaPath(scratch_path / "test_dir", use_absolute=False, ensure_dir=True)
    test_dir2 = gouda.ensure_dir(scratch_path / "test_dir")
    assert str(test_dir) == str(test_dir2)
    assert os.path.isdir(scratch_path / "test_dir")
    assert test_dir.path == str(scratch_path / "test_dir")

    test_dir_path = gouda.ensure_dir(scratch_path / "test_dir", "check1")
    # Have to use test_dir.path. os.path.join uses the __fspath__, which is always absolute
    assert test_dir_path == os.path.join(test_dir.path, "check1")
    assert os.path.isdir(test_dir_path)

    pathlib.Path(os.path.join(test_dir_path, "check2")).touch()
    with pytest.raises(ValueError):
        assert gouda.ensure_dir(test_dir_path, "check2")


def test_next_filename(scratch_path):
    assert gouda.next_filename(scratch_path / "test.txt") == str(scratch_path / "test.txt")
    open(scratch_path / "test.txt", "w").close()
    assert gouda.next_filename(scratch_path / "test.txt") == str(scratch_path / "test_2.txt")
    open(scratch_path / "test_2.txt", "w").close()
    assert gouda.next_filename(scratch_path / "test.txt") == str(scratch_path / "test_3.txt")
    assert gouda.next_filename(scratch_path / "test_2.txt") == str(scratch_path / "test_3.txt")

    assert gouda.next_filename(scratch_path / "test.txt", path_fmt="{idx}{sep}{base_name}{ext}") == str(
        scratch_path / "2_test.txt"
    )
    assert gouda.next_filename(scratch_path / "test.txt", path_fmt="{idx:03}{sep}{base_name}{ext}") == str(
        scratch_path / "002_test.txt"
    )
    assert gouda.next_filename(scratch_path / "test.txt", path_fmt="{base_name}{idx:03}{ext}") == str(
        scratch_path / "test002.txt"
    )


def test_basicname():
    assert gouda.basicname("anypath/morepath/test_item-here.jpg") == "test_item-here"

    head, basename, ext = gouda.fullsplit("anypath/morepath/test_item-here.jpg")
    assert head == "anypath/morepath"
    assert basename == "test_item-here"
    assert ext == ".jpg"

    head, basename, ext = gouda.fullsplit("path/to/file")
    assert head == "path/to"
    assert basename == "file"
    assert ext == ""

    head, basename, ext = gouda.fullsplit("path/to/..file")
    assert head == "path/to"
    assert basename == "..file"
    assert ext == ""

    head, basename, ext = gouda.fullsplit("path/to/..file.txt.gz")
    assert head == "path/to"
    assert basename == "..file"
    assert ext == ".txt.gz"


def test_get_sorted_filenames(scratch_path):
    src_dir = gouda.GoudaPath(scratch_path / "sortedfiles")
    if not src_dir("test.txt").exists():
        gouda.ensure_dir(src_dir)
        for i in range(5):
            with open(gouda.next_filename(src_dir("test.txt")), "w"):
                pass
        for i in range(3):
            with open(gouda.next_filename(src_dir("btest.txt")), "w"):
                pass
        with open(src_dir("btest_extra.txt"), "w"):
            pass
    filenames = gouda.get_sorted_filenames(src_dir / "*.txt")
    assert [gouda.basicname(item) for item in filenames] == [
        "btest",
        "btest_2",
        "btest_3",
        "btest_extra",
        "test",
        "test_2",
        "test_3",
        "test_4",
        "test_5",
    ]


def test_save_load_json_dict(scratch_path):
    temp_data = {"a": 1, "b": 2, "c": 3}
    assert gouda.is_jsonable(temp_data)
    gouda.save_json(temp_data, scratch_path / "test.json")
    assert os.path.isfile(scratch_path / "test.json")
    check_data = gouda.load_json(scratch_path / "test.json")
    for key in temp_data.keys():
        assert temp_data[key] == check_data[key]
    os.remove(scratch_path / "test.json")
    assert not gouda.is_jsonable(os)


def test_save_load_json_list(scratch_path):
    temp_data = ["a", "b", "c"]
    gouda.save_json(temp_data, scratch_path / "test.json")
    assert os.path.isfile(scratch_path / "test.json")
    check_data = gouda.load_json(scratch_path / "test.json")
    for i in range(len(temp_data)):
        assert temp_data[i] == check_data[i]


def test_save_load_json_nested(scratch_path):
    temp_data = [{"a": 1}, {"a": 2}, {"a": 3}]
    gouda.save_json(temp_data, scratch_path / "test.json")
    assert os.path.isfile(scratch_path / "test.json")
    check_data = gouda.load_json(scratch_path / "test.json")
    for i in range(len(temp_data)):
        assert temp_data[i]["a"] == check_data[i]["a"]


def test_save_json_load_json_numpy(scratch_path):
    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, scratch_path / "testx.json", embed_arrays=True, compressed=False)
    gouda.save_json(temp_data, scratch_path / "test2.json", embed_arrays=False, compressed=False)
    gouda.save_json(temp_data, scratch_path / "test3.json", embed_arrays=False, compressed=True)

    assert os.path.isfile(scratch_path / "testx.json")
    assert not os.path.isfile(scratch_path / "testx_array.npz")
    assert not os.path.isfile(scratch_path / "testx_arrayzip.npz")
    assert os.path.isfile(scratch_path / "test2.json")
    assert os.path.isfile(scratch_path / "test2_array.npz")
    assert os.path.isfile(scratch_path / "test3.json")
    assert os.path.isfile(scratch_path / "test3_arrayzip.npz")

    with open(scratch_path / "testx.json", "r") as f:
        data = json.load(f)
        assert data[-1] == "numpy_embed"
    with open(scratch_path / "test2.json", "r") as f:
        data = json.load(f)
        assert data[-1] == "numpy"
    with open(scratch_path / "test3.json", "r") as f:
        data = json.load(f)
        assert data[-1] == "numpy_zip"

    check_data = gouda.load_json(scratch_path / "testx.json")
    check_data2 = gouda.load_json(scratch_path / "test2.json")
    check_data3 = gouda.load_json(scratch_path / "test3.json")
    np.testing.assert_array_equal(temp_data, check_data)
    np.testing.assert_array_equal(temp_data, check_data2)
    np.testing.assert_array_equal(temp_data, check_data3)


def test_save_load_json_set(scratch_path):
    test_data = {"a": set([1, 2, 3])}
    gouda.save_json(test_data, scratch_path / "testset.json")
    check = gouda.load_json(scratch_path / "testset.json")
    assert check["a"] == set([1, 2, 3])


def test_save_load_json_list_numpy(scratch_path):
    temp_data = np.arange(5, dtype=np.uint8)

    gouda.save_json([temp_data], scratch_path / "testn1.json", embed_arrays=False, compressed=False)
    gouda.save_json([temp_data], scratch_path / "testn2.json", embed_arrays=True, compressed=False)
    gouda.save_json([temp_data], scratch_path / "testn3.json", embed_arrays=False, compressed=True)

    check1 = gouda.load_json(scratch_path / "testn1.json")
    check2 = gouda.load_json(scratch_path / "testn2.json")
    check3 = gouda.load_json(scratch_path / "testn3.json")
    assert isinstance(check1, list)
    assert isinstance(check2, list)
    assert isinstance(check3, list)

    np.testing.assert_array_equal(temp_data, check1[0])
    np.testing.assert_array_equal(temp_data, check2[0])
    np.testing.assert_array_equal(temp_data, check3[0])


def test_save_load_json_numpy_list(scratch_path):
    temp_data = [np.arange(3, dtype=np.uint8), np.arange(4, 6, dtype=np.float32)]
    gouda.save_json(temp_data, scratch_path / "testnl.json", embed_arrays=True, compressed=False)
    check_data = gouda.load_json(scratch_path / "testnl.json")
    assert len(check_data) == 2
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    np.testing.assert_array_equal(temp_data[1], check_data[1])

    gouda.save_json(temp_data, scratch_path / "testnl2.json", embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json(scratch_path / "testnl2.json")
    assert len(check_data2) == 2
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    np.testing.assert_array_equal(temp_data[1], check_data2[1])


def test_save_load_json_numpy_dict(scratch_path):
    temp_data = {"a": np.arange(3, dtype=np.uint8), "b": [3, 10]}
    gouda.save_json(temp_data, scratch_path / "testnd.json", embed_arrays=True, compressed=False)
    check_data = gouda.load_json(scratch_path / "testnd.json")
    assert len(temp_data) == len(check_data)
    np.testing.assert_array_equal(temp_data["a"], check_data["a"])
    assert temp_data["b"] == check_data["b"]

    gouda.save_json(temp_data, scratch_path / "testnd2.json", embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json(scratch_path / "testnd2.json")
    assert len(temp_data) == len(check_data2)
    np.testing.assert_array_equal(temp_data["a"], check_data2["a"])


def test_save_load_json_numpy_mixed(scratch_path):
    temp_data = [np.arange(3), 3]
    gouda.save_json(temp_data, scratch_path / "testm.json", embed_arrays=True, compressed=False)
    check_data = gouda.load_json(scratch_path / "testm.json")
    np.testing.assert_array_equal(temp_data[0], check_data[0])
    assert check_data[1] == 3

    gouda.save_json(temp_data, scratch_path / "testm2.json", embed_arrays=False, compressed=False)
    check_data2 = gouda.load_json(scratch_path / "testm2.json")
    np.testing.assert_array_equal(temp_data[0], check_data2[0])
    assert check_data2[1] == 3

    data = [np.int64(32), "a", np.float32(18.32)]
    gouda.save_json(data, scratch_path / "testm3.json")
    check = gouda.load_json(scratch_path / "testm3.json")
    assert check[0] == 32
    assert np.dtype(check[0]) == "int64"
    assert data[1] == "a"
    assert isinstance(data[1], str)
    np.testing.assert_almost_equal(check[2], 18.32, decimal=5)
    assert np.dtype(check[2]) == "float32"

    bad_data = [np.int64(32), "a", np.complex64(18.32 + 3j)]
    with pytest.raises(ValueError):
        gouda.save_json(bad_data, scratch_path / "error.json")


def test_save_json_warning_error(scratch_path):
    temp_data = [1, 2, 3]
    with pytest.warns(UserWarning):
        gouda.save_json(temp_data, scratch_path / "testw.json", embed_arrays=True, compressed=True)
    os.remove(scratch_path / "testw.json")

    temp_data = np.arange(5, dtype=np.uint8)
    gouda.save_json(temp_data, scratch_path / "testw.json", embed_arrays=False)
    gouda.save_json(temp_data.astype(np.float32), scratch_path / "testw2.json", embed_arrays=False)
    gouda.save_json(temp_data.reshape([1, 5]), scratch_path / "testw3.json", embed_arrays=False)
    os.remove(scratch_path / "testw_array.npz")
    os.rename(scratch_path / "testw2_array.npz", scratch_path / "testw_array.npz")
    with pytest.raises(ValueError):
        gouda.load_json(scratch_path / "testw.json")
    os.remove(scratch_path / "testw_array.npz")
    os.rename(scratch_path / "testw3_array.npz", scratch_path / "testw_array.npz")
    with pytest.raises(ValueError):
        gouda.load_json(scratch_path / "testw.json")


def test_is_image(scratch_path):
    test_image = np.ones([20, 20])
    gouda.image.imwrite(scratch_path / "test_image.png", test_image)
    assert gouda.is_image(scratch_path / "test_image.png")
    assert gouda.is_image(scratch_path / ".") is False
    with pytest.raises(FileNotFoundError):
        assert gouda.is_image(scratch_path / "asdfhaksdfhklasjdhfakdhsfk.asdf")
    path = gouda.GoudaPath(scratch_path / "test_image.png")
    assert gouda.is_image(path)


def test_save_load_json_slice(scratch_path):
    data = slice(0, 10, None)
    gouda.save_json(data, scratch_path / "tests.json")
    compare = gouda.load_json(scratch_path / "tests.json")
    assert compare == data
    os.remove(scratch_path / "tests.json")

    data = slice(0, 10, 2)
    gouda.save_json(data, scratch_path / "tests.json")
    compare = gouda.load_json(scratch_path / "tests.json")
    assert compare == data
    os.remove(scratch_path / "tests.json")

    data = [slice(0, 10, 2)]
    gouda.save_json(data, scratch_path / "tests.json")
    compare = gouda.load_json(scratch_path / "tests.json")
    print(compare, data)
    assert compare == data
    os.remove(scratch_path / "tests.json")

    data = [slice(0, 10, 2), {"this": slice(100, 230, None), "that": np.array([1, 2, 3])}]
    gouda.save_json(data, scratch_path / "tests.json")
    compare = gouda.load_json(scratch_path / "tests.json")
    assert compare[0] == slice(0, 10, 2)
    assert isinstance(compare[1], dict)
    assert compare[1]["this"] == slice(100, 230)
    np.testing.assert_array_equal(compare[1]["that"], np.array([1, 2, 3]))
    os.remove(scratch_path / "tests.json")

    data = {
        "size": np.array([500, 500, 500]),
        "origin": np.array([0, 0, 0]),
        "spacing": np.array([1, 1, 1]),
        "dtype": "float32",
        "bounds": (slice(0, 100, None), slice(0, 50)),
    }
    # NOTE - JSON saves tuples and lists as Arrays, so they'll always be loaded as lists
    gouda.save_json(data, scratch_path / "tests.json")
    compare = gouda.load_json(scratch_path / "tests.json")
    for key in data:
        assert key in compare
        if isinstance(data[key], (list, tuple)):
            assert isinstance(compare[key], (list, tuple))
        else:
            assert isinstance(data[key], type(compare[key]))
        if isinstance(data[key], np.ndarray):
            np.testing.assert_allclose(data[key], compare[key])
        elif isinstance(data[key], (list, tuple)):
            assert list(data[key]) == list(compare[key])
        else:
            assert data[key] == compare[key]


def test_read_save_arr(scratch_path):
    x = np.random.randint(-10, 10, [10, 10])
    gouda.save_arr(scratch_path / "test_arr.npy", x)
    assert os.path.exists(scratch_path / "test_arr.npy")
    x2 = gouda.read_arr(scratch_path / "test_arr.npy")
    np.testing.assert_array_equal(x, x2)

    gouda.save_arr(scratch_path / "test_arr2.npy.gz", x)
    assert os.path.exists(scratch_path / "test_arr2.npy.gz")
    assert not os.path.exists(scratch_path / "test_arr2.npy")
    x2 = gouda.read_arr(scratch_path / "test_arr2.npy.gz")
    np.testing.assert_array_equal(x, x2)


def test_fast_glob(scratch_path):
    base_dir = gouda.GoudaPath(scratch_path)("fast_glob").ensure_dir()
    for i in range(10):
        base_dir(f"{i}.txt").touch()
    sub_dir = base_dir("sub_dir").ensure_dir()
    for i in range(3):
        sub_dir(f"sub_{i}.txt").touch()

    check1 = gouda.fast_glob(base_dir, "*.txt", recursive=True)
    assert len(check1) == 13
    assert all([item.endswith(".txt") for item in check1])
    check2 = gouda.fast_glob(base_dir, "*.txt", recursive=False)
    assert len(check2) == 10
    assert set(check2).issubset(set(check1))
    assert all(["sub_dir" not in item for item in check2])

    check_pattern = re.compile(r".*(\.txt)", re.I)
    check3 = gouda.fast_glob(base_dir, check_pattern, recursive=True)
    assert len(check3) == 13
    assert set(check3) == set(check1)

    check4 = gouda.fast_glob(base_dir, check_pattern, sort=True, basenames=True, recursive=False)
    assert len(check4) == 10
    assert all([os.path.sep not in item for item in check4])
    assert set(check4) == set([os.path.basename(item) for item in check2])
    dummy = [item for item in check4]
    dummy.sort(key=lambda x: gouda.basicname(x))
    for a, b in zip(check4, dummy):
        assert a == b


def test_find_images(scratch_path):
    base_dir = gouda.GoudaPath(scratch_path)("fast_glob").ensure_dir()
    for i in range(3):
        base_dir(f"{i}.jpg").touch()
    for i in range(3):
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gouda.image.imwrite(base_dir(f"{i}.png"), test_image)
    sub_dir = base_dir("sub_dir").ensure_dir()
    for i in range(3):
        sub_dir(f"sub_{i}.jpg").touch()
    for i in range(2):
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gouda.image.imwrite(sub_dir(f"sub_{i}.png"), test_image)
    base_dir("distraction").ensure_dir()

    images_1 = gouda.find_images(base_dir, sort=False, basenames=False, recursive=True, fast_check=True)
    assert len(images_1) == 11
    images_2 = gouda.find_images(base_dir, sort=False, basenames=False, recursive=True, fast_check=False)
    assert len(images_2) == 5
    assert set(images_2).issubset(set(images_1))
    assert all([item.endswith(".png") for item in images_2])

    images_3 = gouda.find_images(base_dir, sort=True, basenames=True, recursive=False, fast_check=True)
    assert len(images_3) == 6
    assert all([os.path.sep not in item for item in images_3])
    dummy = [item for item in images_3]
    dummy.sort(key=lambda x: gouda.basicname(x))
    for a, b in zip(images_3, dummy):
        assert a == b

    images_4 = gouda.find_images(base_dir, sort=True, basenames=True, recursive=False, fast_check=False)
    assert len(images_4) == 3
    assert set(images_4).issubset(set(images_3))
