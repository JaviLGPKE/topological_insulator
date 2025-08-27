Topological Insulator
------------

Tight-Binding approximation, for 2D structures, which includes **nearest neighbour hopping**, **spin orbit coupling** and **mean-field interaction** Hamiltonian terms. 
As well as featuring the ability to compute the Chern invariant, Z2 invariant and (L)DOS. Examples are provided within the repository.

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
