import pytest

import numpy as np
import os
import pathlib

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
    test_rel = test_rel.resolve_links()
    assert test_rel.path == test_rel.realpath


def test_representations():
    test_abs = gouda.GoudaPath('absolute', use_absolute=True)
    test_rel = gouda.GoudaPath('relative', use_absolute=False)

    assert str(test_abs) == os.path.abspath('absolute')
    assert str(test_rel) == 'relative'

    assert repr(test_abs) == "GoudaPath('{}')".format(os.path.abspath('absolute'))
    assert repr(test_rel) == "GoudaPath('{}')".format('relative')

    assert os.fspath(test_abs) == os.path.abspath('absolute')
    assert os.fspath(test_rel) == os.fspath('relative')

    assert test_abs.add_basename(test_abs).path == test_abs('absolute').path


def test_navigation():
    test_rel = gouda.GoudaPath('relative', use_absolute=False)
    assert test_rel().path == test_rel.basename()
    assert (test_rel / 'check').path == os.path.join('relative', 'check')
    assert (test_rel // 'check').path == os.path.abspath(os.path.join('relative', 'check'))
    assert (test_rel + 'check').path == test_rel.path + 'check'
    assert (test_rel / 'check')[1] == 'check'
    flavour = pathlib._windows_flavour if os.name == 'nt' else pathlib._posix_flavour
    check_parts = flavour.parse_parts((os.path.abspath(os.path.join('relative', 'check')),))[-1]
    assert (test_rel // 'check')[1] == check_parts[1]
    assert flavour.parse_parts(((test_rel // 'check')[:3],))[-1] == check_parts[:3]
    assert len(test_rel) == 1
    assert len(test_rel(use_absolute=True)) == len(flavour.parse_parts((os.path.abspath('relative'),))[-1])

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
    test_dir('.hidden.txt').touch()
    assert test_dir('.hidden.txt').is_hidden()
    with pytest.raises(FileNotFoundError):
        test_dir('.fake_file.txt').is_hidden()
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

    assert test_dir.num_children(dirs_only=False, files_only=False, include_hidden=True) == 4
    assert test_dir.num_children(dirs_only=True, files_only=False, include_hidden=False) == 2
    assert test_dir.num_children(dirs_only=False, files_only=True, include_hidden=True) == 2

    with pytest.raises(NotADirectoryError):
        assert test_dir('check_file.txt').children()
    with pytest.raises(NotADirectoryError):
        for item in test_dir('check_file.txt').iterdir():
            pass

    globbed = test_dir.glob('*')
    assert len(globbed) == 3
    assert test_dir('check_dir1').path in globbed

    first_glob = test_dir.globfirst('*')
    assert first_glob == globbed[0]
    first_glob = test_dir.globfirst('*', as_gouda=True)
    assert isinstance(first_glob, gouda.GoudaPath)
    assert first_glob.path == globbed[0]
    first_glob = test_dir.globfirst('*', basename=True)
    assert first_glob == os.path.basename(globbed[0])
    assert test_dir.globfirst('aksjdfhakljsdfhklajsdhfkjash3') is None

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
    os.remove(test_dir / '.hidden.txt')
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
    assert path.stem() == 'test'

    path = gouda.GoudaPath('path/to/file/test.nii.gz')
    assert path.stem() == 'test'
    assert path.extension() == '.nii.gz'

    path = gouda.GoudaPath('path/to/file/..code.txt')
    assert path.stem() == '..code'
    assert path.extension() == '.txt'

    path = gouda.GoudaPath('path/to/file/..code')
    assert path.stem() == '..code'
    assert path.extension() == ''


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


def test_insert_compare():
    test_path = gouda.GoudaPath('path/to/file.txt', use_absolute=False)
    assert test_path == 'path/to/file.txt'
    alt_test_path = gouda.GoudaPath('path/to/file.txt', use_absolute=False)
    assert test_path == alt_test_path
    assert test_path <= alt_test_path
    assert alt_test_path >= test_path
    assert test_path == 'path/to/file.txt'
    assert test_path <= 'path/to/file.txt'
    assert 'path/to/file.txt' >= test_path

    test_path[1] = 'from'
    assert test_path == 'path/from/file.txt'
    assert test_path != alt_test_path
    assert test_path < alt_test_path
    assert test_path <= alt_test_path
    assert alt_test_path > test_path
    assert alt_test_path >= test_path
    assert test_path != 'path/to/file.txt'
    assert test_path < 'path/to/file.txt'
    assert test_path <= 'path/to/file.txt'
    assert 'path/to/file.txt' > test_path
    assert 'path/to/file.txt' >= test_path
    assert gouda.GoudaPath('path/to/file.txt') > 'path/from/file.txt'
    assert gouda.GoudaPath('path/to/file.txt') >= 'path/from/file.txt'

    test_path.insert(2, 'this')
    assert test_path == 'path/from/this/file.txt'

    assert test_path.with_stem('other_file') == 'path/from/this/other_file.txt'
    assert test_path.with_basename('other_file.csv') == 'path/from/this/other_file.csv'
    assert test_path.with_extension('.csv') == 'path/from/this/file.csv'
    assert test_path.with_extension('csv') == 'path/from/this/file.csv'
    assert isinstance(test_path.as_pathlib(), type(pathlib.Path()))
    assert test_path.as_pathlib() == pathlib.Path('path/from/this/file.txt')

    test_path2 = gouda.GoudaPath('test/file.tar.gz')
    assert test_path2.with_extension('csv') == 'test/file.csv'
    assert test_path2.with_stem('other_file') == 'test/other_file.tar.gz'
    assert test_path2.with_basename('other_file.csv') == 'test/other_file.csv'

    windows_test_path = gouda.GoudaPath(r'C:\\Windows\test\path.txt')
    windows_test_path.flavour = pathlib._windows_flavour
    windows_test_path._clear_cache()
    print(windows_test_path.as_posix())
    assert windows_test_path.as_posix() == 'C://Windows/test/path.txt'

    assert gouda.GoudaPath.cwd() == os.getcwd()
    assert gouda.GoudaPath('~/test').expanduser() == os.path.expanduser('~/test')


def test_readwrite():
    test_dir = gouda.GoudaPath('ScratchFiles/goudapath_readwrite_directory').ensure_dir()
    with open(test_dir / 'test.txt', 'w') as f:
        f.write('test\n')
        f.write('test_line2')
    with open(test_dir / 'test.txt', 'r') as f:
        assert f.read() == 'test\ntest_line2'
    with open(test_dir / 'test.txt', 'r') as f:
        assert f.readlines() == ['test\n', 'test_line2']
    test_dir('test2.txt').write_text('here is a test')
    with open(test_dir('test2.txt')) as f:
        assert f.read() == 'here is a test'
    assert test_dir('test2.txt').read_text() == 'here is a test'
    with pytest.raises(TypeError):
        test_dir('test2.txt').write_text(b'bytes test')

    test_bytes = b'bytes_test'
    with open(test_dir('test3.bin'), 'wb') as f:
        f.write(memoryview(test_bytes))
    with open(test_dir('test3.bin'), 'rb') as f:
        assert f.read() == b'bytes_test'
    test_dir('test4.bin').write_bytes(b'more bytes test')
    assert test_dir('test4.bin').read_bytes() == b'more bytes test'

    with pytest.raises(FileExistsError):
        test_dir('test4.bin').touch(exist_ok=False)

    os.remove(test_dir / 'test.txt')
    os.remove(test_dir / 'test2.txt')
    os.remove(test_dir / 'test3.bin')
    os.remove(test_dir / 'test4.bin')
    os.removedirs(test_dir)
