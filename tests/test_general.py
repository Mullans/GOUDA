from __future__ import annotations

import pytest

import os

import gouda


def test_hasgetattr_recursive():
    assert gouda.hasattr_recursive(os, "path")
    assert gouda.hasattr_recursive(os, "path.basename")
    assert not gouda.hasattr_recursive(os, "asdf")
    assert not gouda.hasattr_recursive(os, "path.asdf")

    check = gouda.getattr_recursive(os, "path.basename")
    assert check == os.path.basename
    assert check != os.path

    with pytest.raises(AttributeError):
        check = gouda.getattr_recursive(os, "asdf")
    with pytest.raises(AttributeError):
        check = gouda.getattr_recursive(os, "path.asdf")


def test_nestit():
    iter1 = "abc"
    expected = "abc"
    for idx, val in enumerate(gouda.nestit(iter1)):
        assert val == [expected[idx]], f"{val} is not equal to {expected[idx]}"

    iter2 = [1, 2, 3]
    expected = [["a", 1], ["a", 2], ["a", 3], ["b", 1], ["b", 2], ["b", 3], ["c", 1], ["c", 2], ["c", 3]]
    for idx, val in enumerate(gouda.nestit(iter1, iter2)):
        assert val == expected[idx], f"{val} is not equal to {expected[idx]}"

    iter3 = tuple(["alphabet"])
    expected = [
        ["a", 1, "alphabet"],
        ["a", 2, "alphabet"],
        ["a", 3, "alphabet"],
        ["b", 1, "alphabet"],
        ["b", 2, "alphabet"],
        ["b", 3, "alphabet"],
        ["c", 1, "alphabet"],
        ["c", 2, "alphabet"],
        ["c", 3, "alphabet"],
    ]
    for idx, val in enumerate(gouda.nestit(iter1, iter2, iter3)):
        assert val == expected[idx], f"{val} is not equal to {expected[idx]}"

    iter_end = gouda.nestit("abc")
    assert next(iter_end) == ["a"]
    assert next(iter_end) == ["b"]
    assert next(iter_end) == ["c"]
    with pytest.raises(StopIteration):
        next(iter_end)

    empty_iter = gouda.nestit("")
    with pytest.raises(ValueError):
        next(empty_iter)

    test_cycle = gouda.capped_cycle([])
    assert next(test_cycle) == StopIteration


def test_isiter():
    assert not gouda.is_iter("x")
    assert gouda.is_iter([1, 2, 3])
    assert not gouda.is_iter(1)


def test_len():
    x = 3
    assert gouda.force_len(x, 3) == (3, 3, 3)
    assert gouda.force_len(x, 2) == (3, 3)
    assert gouda.force_len((3, 2), 2) == (3, 2)
    assert gouda.force_len((3, 2), 3, pad="wrap") == (3, 2, 3)
    assert gouda.force_len((3, 2), 3, pad="reflect") == (3, 2, 2)
    assert gouda.force_len((3, 2), 4, pad="reflect") == (3, 2, 2, 3)
    assert gouda.force_len((1, 2, 3, 4), 2) == (1, 2)
    with pytest.raises(ValueError):
        assert gouda.force_len((3, 2), 5, pad="reflect")
    with pytest.raises(ValueError):
        assert gouda.force_len((3, 2), 5, pad="asdf")

    a = [1, 2]
    b = 2
    c = 3
    result, result_len = gouda.match_len(a, b, c)
    assert all([len(x) == result_len for x in result])
    assert result_len == 2
    result, result_len = gouda.match_len(a, b, c, count=4, pad="wrap")
    assert result_len == 4
    assert all([len(x) == 4 for x in result])
