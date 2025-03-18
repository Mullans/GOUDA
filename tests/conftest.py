# -*- coding: utf-8 -*-
"""
Dummy conftest.py for gouda.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
https://pytest.org/latest/plugins.html
"""

from __future__ import annotations


import pytest


@pytest.fixture(scope="session")
def scratch_path(tmp_path_factory):
    return tmp_path_factory.mktemp("ScratchFiles")
