Topological Insulator
------------

Tight-Binding approximation which includes **nearest neighbour hopping** and **spin orbit coupling** Hamiltonian terms.

Installation
------------

To use the library in a Linux environment, simply install it using pip:

.. code-block:: console

   $ git clone https://github.com/JaviLGPKE/topological_insulator.git
   $ cd topological_insulator
   $ pip install -e .

.. If your *PYTHONPATH* doesn't contains pybind11, you should add it:

.. .. code-block:: console

..    $ export pybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")