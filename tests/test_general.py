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


def test_iter_batch():
    """Test the iter_batch function with various input types and edge cases."""
    
    # Test with list input
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    batches = list(gouda.iter_batch(data, 3))
    expected = [(1, 2, 3), (4, 5, 6), (7, 8)]
    assert batches == expected
    
    # Test with tuple input
    data = (1, 2, 3, 4, 5)
    batches = list(gouda.iter_batch(data, 2))
    expected = [(1, 2), (3, 4), (5,)]
    assert batches == expected
    
    # Test with string input (iterable of characters)
    data = "abcdef"
    batches = list(gouda.iter_batch(data, 2))
    expected = [('a', 'b'), ('c', 'd'), ('e', 'f')]
    assert batches == expected
    
    # Test with range input
    data = range(10)
    batches = list(gouda.iter_batch(data, 4))
    expected = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert batches == expected
    
    # Test with empty iterable
    data = []
    batches = list(gouda.iter_batch(data, 3))
    expected = []
    assert batches == expected
    
    # Test with single element
    data = [42]
    batches = list(gouda.iter_batch(data, 3))
    expected = [(42,)]
    assert batches == expected
    
    # Test with batch_size of 1
    data = [1, 2, 3, 4]
    batches = list(gouda.iter_batch(data, 1))
    expected = [(1,), (2,), (3,), (4,)]
    assert batches == expected
    
    # Test with batch_size larger than iterable
    data = [1, 2, 3]
    batches = list(gouda.iter_batch(data, 5))
    expected = [(1, 2, 3)]
    assert batches == expected
    
    # Test with batch_size equal to iterable length
    data = [1, 2, 3, 4]
    batches = list(gouda.iter_batch(data, 4))
    expected = [(1, 2, 3, 4)]
    assert batches == expected
    
    # Test with generator input
    def gen():
        yield 1
        yield 2
        yield 3
        yield 4
    
    batches = list(gouda.iter_batch(gen(), 2))
    expected = [(1, 2), (3, 4)]
    assert batches == expected
    
    # Test that it returns tuples
    data = [1, 2, 3, 4]
    batches = list(gouda.iter_batch(data, 2))
    for batch in batches:
        assert isinstance(batch, tuple)
    
    # Test with mixed types
    data = [1, "hello", 3.14, True, None]
    batches = list(gouda.iter_batch(data, 2))
    expected = [(1, "hello"), (3.14, True), (None,)]
    assert batches == expected
    
    # Test that the function is a generator (doesn't consume all at once)
    data = [1, 2, 3, 4, 5, 6]
    batch_gen = gouda.iter_batch(data, 2)
    
    # Get first batch
    first_batch = next(batch_gen)
    assert first_batch == (1, 2)
    
    # Get second batch
    second_batch = next(batch_gen)
    assert second_batch == (3, 4)
    
    # Get remaining batches
    remaining = list(batch_gen)
    assert remaining == [(5, 6)]
    
    # Test with zero batch_size (should raise error)
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        list(gouda.iter_batch(data, 0))
    
    # Test with negative batch_size (should raise error)
    data = [1, 2, 3]
    with pytest.raises(ValueError):
        list(gouda.iter_batch(data, -1))
