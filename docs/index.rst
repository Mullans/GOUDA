=====
GOUDA
=====

Welcome to the documentation of **GOUDA** (**G** ood **O** ld **U** tilities for **D** ata **A** nalysis).

.. .. note::
..
..     This is the main page of your project's `Sphinx`_ documentation.
..     It is formatted in `reStructuredText`_. Add additional pages
..     by creating rst-files in ``docs`` and adding them to the `toctree`_ below.
..     Use then `references`_ in order to link them from this page, e.g.
..     :ref:`authors` and :ref:`changes`.
..
..     It is also possible to refer to the documentation of other Python packages
..     with the `Python domain syntax`_. By default you can reference the
..     documentation of `Sphinx`_, `Python`_, `NumPy`_, `SciPy`_, `matplotlib`_,
..     `Pandas`_, `Scikit-Learn`_. You can add more by extending the
..     ``intersphinx_mapping`` in your Sphinx's ``conf.py``.
..
..     The pretty useful extension `autodoc`_ is activated by default and lets
..     you include documentation from docstrings. Docstrings can be written in
..     `Google style`_ (recommended!), `NumPy style`_ and `classical style`_.

Requirements
============

* `NumPy`_
* `Matplotlib`_

Optional Requirements (gouda.image only)
========================================
* `OpenCV <https://opencv.org/>`_


Contents
========

.. toctree::
   :maxdepth: 2

   binaryconfusionmatrix
   data_methods
   display
   file_methods
   goudapath
   image
   moving_stats

   License <license>
   Authors <authors>
   Changelog <changelog>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable/
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _matplotlib: https://matplotlib.org/contents.html#
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
   .. _SciPy: http://docs.scipy.org/doc/scipy/reference/
   .. _Pandas: http://pandas.pydata.org/pandas-docs/stable
   .. _Scikit-Learn: http://scikit-learn.org/stable
   .. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
   .. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
