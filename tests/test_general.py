import os

import gouda
import pytest


def test_hasgetattr_recursive():
    assert gouda.hasattr_recursive(os, 'path')
    assert gouda.hasattr_recursive(os, 'path.basename')
    assert not gouda.hasattr_recursive(os, 'asdf')
    assert not gouda.hasattr_recursive(os, 'path.asdf')

    check = gouda.getattr_recursive(os, 'path.basename')
    assert check == os.path.basename
    assert check != os.path

    with pytest.raises(AttributeError):
        check = gouda.getattr_recursive(os, 'asdf')
    with pytest.raises(AttributeError):
        check = gouda.getattr_recursive(os, 'path.asdf')
