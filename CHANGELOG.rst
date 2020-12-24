=========
Changelog
=========

Version 0.1
===========

- Add pre-existing utilities and new tests

Version 0.2
===========

- Add 'allow_resize' option to image.rotate
- Add 'add_array' method to ConfusionMatrix for faster adding of numpy arrays
- Fixed divide by 0 warning in ConfusionMatrix.sensitivity
- Added option to show/hide metrics in ConfusionMatrix.print and fixed formatting large numbers
- Added as_greyscale and unchanged options to imread
- Add standardize/rescale/normalize methods
- Can now pass a list of directories to ensure_dir to be made as part of one path
- Added num_digits
- Changed load_json/save_json to allow for splitting large arrays into a .npz file with placeholder keys in the JSON
- Add gouda.image.polar_to_cartesian
- Add precision and Matthews correlation coefficient (binary only) to ConfusionMatrix


Version 0.3
===========

- Add GoudaPath
- Add GoudaPath support for image reading/writing methods
- Add factorizing Methods
- Add dict_flip to reverse dictionary key/value pairs
- Updated print_grid and added single print_image
- Add BinaryConfusionMatrix
