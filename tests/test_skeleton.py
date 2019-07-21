# -*- coding: utf-8 -*-

import pytest
from gouda.skeleton import fib

__author__ = "Sean Mullan"
__copyright__ = "Sean Mullan"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
