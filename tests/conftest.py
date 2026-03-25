# -*- coding: utf-8 -*-
from __future__ import annotations


import pytest


@pytest.fixture(scope="session")
def scratch_path(tmp_path_factory):
    return tmp_path_factory.mktemp("ScratchFiles")
