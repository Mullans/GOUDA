*****
GOUDA
*****

Requirements
============

* `NumPy <https://github.com/numpy/numpy>`_

.. automodule:: gouda

File Methods
=============
.. autofunction:: ensure_dir
.. autofunction:: next_filename
.. autofunction:: save_json
.. autofunction:: load_json

Data Methods
============
.. autofunction:: arr_sample
.. autofunction:: sigmoid
.. autofunction:: get_specificities
.. autofunction:: get_accuracy
.. autofunction:: get_binary_confusion_matrix
.. autofunction:: get_confusion_matrix

Text Methods
============
.. autofunction:: print_confusion_matrix
.. autofunction:: underline

Statistical Classes
===================
ConfusionMatrix
---------------
.. autoclass:: ConfusionMatrix
   :members:
   :special-members: __add__, __iadd__
   :private-members:

MMean
-----
.. autoclass:: MMean
   :members:
   :special-members: __add__, __iadd__, __str__, __sub__

MStddev
-------
.. autoclass:: MStddev
   :members:
   :special-members: __add__, __iadd__, __str__, __sub__

MMeanArray
----------
.. autoclass:: MMeanArray
   :members:
   :special-members: __add__, __iadd__, __str__, __sub__

MStddevArray
------------
.. autoclass:: MStddevArray
   :members:
   :special-members: __add__, __iadd__, __str__, __sub__
