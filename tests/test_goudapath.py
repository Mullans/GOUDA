import os

import numpy as np
import pytest

import gouda

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
    test_basic = gouda.GoudaPath(use_absolute=False)
    assert test_basic.path == '.'

    test_abs = gouda.GoudaPath('absolute', use_absolute=True)
    test_rel = gouda.GoudaPath('relative', use_absolute=False)

    assert os.path.expanduser('~') in test_abs.path
    assert test_abs.path == os.path.abspath('absolute')
    assert test_rel.path == 'relative'
    assert 'relative' in test_rel
    assert 'relative' not in test_abs
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

    assert test_abs.abspath == test_abs.realpath
    test_rel.resolve_links()
    assert test_rel.path == test_rel.realpath


def test_representations():
    test_abs = gouda.GoudaPath('absolute', use_absolute=True)
    test_rel = gouda.GoudaPath('relative', use_absolute=False)

    assert str(test_abs) == os.path.abspath('absolute')
    assert str(test_rel) == 'relative'

    assert repr(test_abs) == "GoudaPath('{}')".format(os.path.abspath('absolute'))
    assert repr(test_rel) == "GoudaPath('{}')".format('relative')

    assert os.fspath(test_abs) == os.path.abspath('absolute')
    assert os.fspath(test_rel) == os.path.abspath('relative')

    assert test_abs.add_basename(test_abs).path == test_abs('absolute').path


def test_navigation():
    test_rel = gouda.GoudaPath('relative', use_absolute=False)
    assert test_rel().path == test_rel.basename()
    assert (test_rel / 'check').path == os.path.join('relative', 'check')
    assert (test_rel // 'check').path == os.path.abspath(os.path.join('relative', 'check'))
    assert (test_rel + 'check').path == test_rel.path + 'check'
    assert (test_rel / 'check')[1] == 'check'
    assert (test_rel // 'check')[1] == os.path.abspath(os.path.join('relative', 'check'))[1:].split(os.path.sep)[1]
    assert (test_rel // 'check')[1:2] == os.path.sep.join(os.path.abspath(os.path.join('relative', 'check'))[1:].split(os.path.sep)[1:2])
    assert len(test_rel) == 1
    assert len(test_rel(use_absolute=True)) == len(os.path.abspath('relative')[1:].split(os.path.sep))

    multiples = test_rel(['a', 'b', 'c'])
    assert len(multiples) == 3
    assert multiples[0].path == test_rel('a').path
    assert multiples[2].path == test_rel('c').path
    child_check1 = test_rel('a', use_absolute=False)
    child_check2 = test_rel('a', use_absolute=True)
    assert child_check1.path != child_check2.path
    assert child_check1.abspath == child_check2.path


def test_relation():
    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_test_directory')
    gouda.ensure_dir(test_dir)
    gouda.ensure_dir(test_dir / 'check_dir1')
    gouda.ensure_dir(test_dir / 'check_dir2')
    with open(test_dir / 'check_file.txt', 'w') as _:
        pass
    if test_dir('image1.png').exists():
        os.remove(test_dir('image1.png'))
    if test_dir('image2.png').exists():
        os.remove(test_dir('image2.png'))

    assert test_dir('check_dir1').exists()
    assert test_dir('check_dir1').is_dir()
    assert gouda.GoudaPath('goudapath_test_directory', use_absolute=False).parent_dir().path[:2] == os.pardir
    assert test_dir('check_dir1').parent_dir().path == test_dir.path
    assert not test_dir('doesntexist').exists()
    assert test_dir('check_file.txt').extension() == '.txt'
    assert test_dir('check_subdir').extension() == ''

    children = test_dir.children(dirs_only=True, files_only=False, basenames=False)
    assert len(children) == 2
    assert os.path.join(test_dir.path, 'check_dir1') in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_dir2') in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_file.txt') not in [item.path for item in children]

    children = test_dir.children(dirs_only=False, files_only=True, basenames=False)
    assert len(children) == 1
    assert os.path.join(test_dir.path, 'check_dir1') not in [item.path for item in children]
    assert os.path.join(test_dir.path, 'check_file.txt') in [item.path for item in children]

    children = test_dir.children(dirs_only=True, files_only=True, basenames=False)
    assert len(children) == 0

    children = test_dir.children(dirs_only=True, files_only=False, basenames=True, include_hidden=True)
    assert 'check_dir1' in children
    assert 'check_dir2' in children

    assert test_dir.num_children(dirs_only=False, files_only=False, include_hidden=True) == 3
    assert test_dir.num_children(dirs_only=True, files_only=False, include_hidden=False) == 2
    assert test_dir.num_children(dirs_only=False, files_only=True, include_hidden=True) == 1

    with pytest.raises(NotADirectoryError):
        assert test_dir('check_file.txt').children()

    globbed = test_dir.glob('*')
    assert len(globbed) == 3
    assert test_dir('check_dir1').path in globbed

    globbed = test_dir.glob('*.txt', basenames=True)
    assert 'check_file.txt' in globbed

    globbed = test_dir.glob('*', sort=True)
    assert test_dir('check_dir1').path == globbed[0]

    globbed = test_dir.glob('*', sort=True, as_gouda=True)
    assert isinstance(globbed[0], gouda.GoudaPath)

    globbed = test_dir.parent_dir().glob('**/*.txt', recursive=True)
    assert test_dir('check_file.txt').path in globbed

    setitem_tester = test_dir / 'check_file.txt'
    assert setitem_tester[-1] == 'check_file.txt'
    assert setitem_tester[-2] == test_dir.basename()
    setitem_tester[-1] = 'null_file.txt'
    assert setitem_tester[-1] == 'null_file.txt'
    assert setitem_tester[-2] == test_dir.basename()
    setitem_tester2 = setitem_tester.replace('null', 'blank')
    assert setitem_tester2[-1] == 'blank_file.txt'
    setitem_tester3 = gouda.GoudaPath('just/a/test', use_absolute=False)
    setitem_tester3[0] = 'maybe'
    assert setitem_tester3[0] == 'maybe'
    assert setitem_tester3[1] == 'a'
    assert setitem_tester3[2] == 'test'

    os.remove(test_dir / 'check_file.txt')
    os.removedirs(test_dir / 'check_dir1')
    os.removedirs(test_dir / 'check_dir2')


def test_get_images_num_children():
    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_imagetest_directory')
    gouda.ensure_dir(test_dir)
    gouda.ensure_dir(test_dir / 'check_dir1')
    gouda.ensure_dir(test_dir / 'check_dir2')
    with open(test_dir / 'check_file.txt', 'w') as _:
        pass
    if test_dir('image1.png').exists():
        os.remove(test_dir('image1.png'))
    if test_dir('image2.png').exists():
        os.remove(test_dir('image2.png'))

    test_img = np.ones([50, 50, 3])
    gouda.image.imwrite(test_dir / 'image1.png', test_img)
    gouda.image.imwrite(test_dir / 'image2.png', test_img)
    image_results = test_dir.get_images(basenames=True)
    assert 'image1.png' in image_results
    assert 'image2.png' in image_results
    assert 'check_file.txt' not in image_results

    image_results2 = test_dir.get_images(basenames=True, fast_check=True)
    image_results3 = test_dir.get_images(basenames=False)
    for a, b, c in zip(image_results, image_results2, image_results3):
        assert a == b
        assert a == os.path.basename(c)

    image_results4 = test_dir.get_images(basenames=True, sort=True)
    image_results5 = test_dir.get_images(basenames=False, sort=True)
    for a, b in zip(image_results4, image_results5):
        assert a == os.path.basename(b)
    assert image_results4[0] == 'image1.png'
    assert image_results4[1] == 'image2.png'

    assert test_dir('image1.png').is_image()
    with pytest.raises(NotADirectoryError):
        assert test_dir('image1.png').num_children()
    with pytest.raises(NotADirectoryError):
        assert test_dir('image1.png').get_images()
    assert test_dir.is_image() is False
    assert test_dir.num_children(dirs_only=True, files_only=False, include_hidden=True) == 2
    assert test_dir.num_children(dirs_only=False, files_only=True, include_hidden=False) == 3
    os.remove(test_dir / 'image1.png')
    os.remove(test_dir / 'image2.png')
    os.removedirs(test_dir / 'check_dir1')
    os.removedirs(test_dir / 'check_dir2')


def test_encode():
    path = gouda.GoudaPath('.')
    assert os.path.abspath('.').encode() == path.__bytes__()


def test_strings():
    path = gouda.GoudaPath('testerpath', use_absolute=False)
    assert path.startswith('tester')
    assert path.endswith('path')

    path = gouda.GoudaPath('testerpath', use_absolute=True)
    assert path.startswith('tester') is False
    assert path.endswith('path')

    path = gouda.GoudaPath('test  ', use_absolute=False)
    assert path.rstrip().path == 'test'
    assert path.rstrip('.').path == path.path

    path = gouda.GoudaPath('test.txt')
    assert path[-1] == 'test.txt'
    assert path.basename() == 'test.txt'
    assert path.basicname() == 'test'


def test_ensure_dir():
    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_ensuretest_directory')
    gouda.ensure_dir(test_dir)
    for check_dir in ['check_dir', 'check_dir2', 'check_dir3', 'check_dir4']:
        if os.path.exists(f'ScratchFiles/goudapath_ensuretest_directory/{check_dir}'):
            os.removedirs(f'ScratchFiles/goudapath_ensuretest_directory/{check_dir}')

    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_ensuretest_directory/check_dir')
    gouda.ensure_dir(test_dir)
    assert os.path.exists(test_dir)
    assert test_dir.exists()

    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_ensuretest_directory/check_dir2', ensure_dir=True)
    assert os.path.exists(test_dir)
    assert test_dir.exists()

    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_ensuretest_directory/check_dir3')
    test_dir2 = test_dir.ensure_dir()
    assert os.path.exists(test_dir)
    assert test_dir.exists()
    assert str(test_dir) == str(test_dir2)

    test_file = gouda.GoudaPath('ScratchFiles/goudapath_ensuretest_directory/check_dir4/nothing.txt', ensure_dir=True)
    assert not os.path.exists(test_file)
    assert os.path.exists(test_file.parent_dir())

    gouda.ensure_dir(test_dir)
    for check_dir in ['check_dir', 'check_dir2', 'check_dir3', 'check_dir4']:
        if os.path.exists(f'ScratchFiles/goudapath_ensuretest_directory/{check_dir}'):
            os.removedirs(f'ScratchFiles/goudapath_ensuretest_directory/{check_dir}')
