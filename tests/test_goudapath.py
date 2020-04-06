import os

import pytest

from gouda import GoudaPath

# def test_ensure_dir():
#     test_dir = gouda.ensure_dir("test_dir")
#     test_dir2 = gouda.ensure_dir("test_dir")
#     assert test_dir == test_dir2
#     assert os.path.isdir("test_dir")
#     assert test_dir == "test_dir"
#
#     test_dir_path = gouda.ensure_dir("test_dir", "check1")
#     assert test_dir_path == os.path.join(test_dir, "check1")
#     assert os.path.isdir(test_dir_path)
#
#     pathlib.Path(os.path.join(test_dir_path, 'check2')).touch()
#     with pytest.raises(ValueError):
#         assert gouda.ensure_dir(test_dir_path, "check2")
#
#     # Cleanup
#     os.remove(os.path.join(test_dir_path, 'check2'))
#     os.rmdir(test_dir_path)
#     os.rmdir('test_dir')


def test_init_call():
    test_abs = GoudaPath('absolute', use_absolute=True)
    test_rel = GoudaPath('relative', use_absolute=False)

    assert os.path.expanduser('~') in test_abs.path
    assert test_abs.path == os.path.abspath('absolute')
    assert test_rel.path == 'relative'
    assert test_abs.abspath == os.path.abspath('absolute')
    assert test_rel.abspath == os.path.abspath('relative')
    assert test_abs.basename() == 'absolute'
    assert test_rel.basename() == 'relative'
    assert test_abs.use_absolute
    assert not test_rel.use_absolute

    assert test_abs().path == os.path.abspath('absolute')
    assert test_rel().path == 'relative'
    assert test_abs(use_absolute=False).path == os.path.abspath('absolute')
    assert test_rel(use_absolute=True).path == os.path.abspath('relative')
    assert test_abs(use_absolute=True).path == os.path.abspath('absolute')
    assert test_rel(use_absolute=False).path == test_rel().path

    file_list = ['check1', 'check2', 'check3']
    assert [item.path for item in test_abs(file_list)] == [os.path.join(os.path.abspath('absolute'), item) for item in file_list]
    assert [item.path for item in test_rel(file_list)] == [os.path.join('relative', item) for item in file_list]
    assert [item.path for item in test_abs(*file_list)] == [os.path.join(os.path.abspath('absolute'), item) for item in file_list]
    assert [item.path for item in test_rel(*file_list)] == [os.path.join('relative', item) for item in file_list]


def test_representations():
    test_abs = GoudaPath('absolute', use_absolute=True)
    test_rel = GoudaPath('relative', use_absolute=False)

    assert str(test_abs) == os.path.abspath('absolute')
    assert str(test_rel) == 'relative'

    assert repr(test_abs) == "GoudaPath('{}')".format(os.path.abspath('absolute'))
    assert repr(test_rel) == "GoudaPath('{}')".format('relative')

    assert os.fspath(test_abs) == os.path.abspath('absolute')
    assert os.fspath(test_rel) == 'relative'


def test_navigation():
    test_rel = GoudaPath('relative', use_absolute=False)
    assert (test_rel / 'check').path == os.path.join('relative', 'check')
    assert (test_rel // 'check').path == os.path.abspath(os.path.join('relative', 'check'))
    assert (test_rel + 'check').path == test_rel.path + 'check'
    assert (test_rel / 'check')[1] == 'check'
    assert (test_rel // 'check')[1] == os.path.abspath(os.path.join('relative', 'check'))[1:].split(os.path.sep)[1]
    assert (test_rel // 'check')[1:2] == os.path.sep.join(os.path.abspath(os.path.join('relative', 'check'))[1:].split(os.path.sep)[1:2])
    assert len(test_rel) == 1
    assert len(test_rel(use_absolute=True)) == len(os.path.abspath('relative')[1:].split(os.path.sep))


def test_relation():
    test_dir = GoudaPath('goudapath_test_directory')
    os.mkdir(test_dir)
    os.mkdir(test_dir / 'check_dir1')
    os.mkdir(test_dir / 'check_dir2')
    with open(test_dir / 'check_file.txt', 'w') as _:
        pass

    assert test_dir('check_dir1').exists()
    assert test_dir('check_dir1').is_dir()
    assert GoudaPath('goudapath_test_directory', use_absolute=False).parent_dir().path[:2] == os.pardir
    assert test_dir('check_dir1').parent_dir().path == test_dir.path
    assert not test_dir('doesntexist').exists()
    assert test_dir('check_file.txt').extension() == '.txt'

    children = test_dir.children(dirs_only=True, exclude_dirs=False, basenames=False)
    assert len(children) == 2
    assert os.path.join(test_dir.path, 'check_dir1') in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_dir2') in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_file.txt') not in [item.path for item in children]

    children = test_dir.children(dirs_only=False, exclude_dirs=True, basenames=False)
    assert len(children) == 1
    assert os.path.join(test_dir.path, 'check_dir1') not in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_file.txt') in [item.path for item in children]

    children = test_dir.children(dirs_only=True, exclude_dirs=True, basenames=False)
    assert len(children) == 0

    children = test_dir.children(dirs_only=True, exclude_dirs=False, basenames=True)
    assert 'check_dir1' in children
    assert 'check_dir2' in children

    with pytest.raises(NotADirectoryError):
        assert test_dir('check_file.txt').children()

    globbed = test_dir.glob('*')
    assert len(globbed) == 3
    assert test_dir('check_dir1').path in globbed

    globbed = test_dir.glob('*.txt', basenames=True)
    assert 'check_file.txt' in globbed

    globbed = test_dir.glob('*', sort=True)
    assert test_dir('check_dir1').path == globbed[0]

    globbed = test_dir.parent_dir().glob('**/*.txt', recursive=True)
    assert test_dir('check_file.txt').path in globbed

    os.remove(test_dir / 'check_file.txt')
    os.removedirs(test_dir / 'check_dir1')
    os.removedirs(test_dir / 'check_dir2')
