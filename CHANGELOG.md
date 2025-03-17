# Change log

## [Unreleased]

### Added

### Changed

* `gouda.display.print_image`, `gouda.display.show_video`, and `gouda.display.print_grid` now use `file_name` instead of `toFile` to specify the optional output file name. The `toFile` argument has been removed.
* `gouda.file_methods.fast_glob` and `gouda.file_methods.find_images` now use `as_iterator` instead of `iter`.
* `gouda.gouda_path.GoudaPath.get_images` and `gouda.gouda_path.GoudaPath.search` now use `as_iterator` instead of `iter`.
* `gouda.image.get_bounds` now returns a list of tuples instead of a list of lists.
    * This also affects `gouda.image.crop_to_content`

### Deprecated

### Removed

* `gouda.image.horizontal_flip` and `gouda.image.vertical_flip` - These were just wrappers for `x[:, ::-1]` and `x[::-1]` respectively.
