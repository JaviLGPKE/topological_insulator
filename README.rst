Topological Insulators
------------

Tight-Binding approximation which includes **nearest neighbour hopping** and **spin orbit coupling** Hamiltonian terms.

Installation
------------

The **Solver C++** uses the finite element platform FEniCSx and a better, custom made version of multiphenicsx. 
It is assumed you have a working conda environment or likewise.

To use the library, first install it using pip:

.. code-block:: console

   $ git clone https://github.com/JaviLGPKE/topological_insulator.git
   $ cd topological_insulator
   $ pip install -e .

If your *PYTHONPATH* doesn't contains pybind11, you should add it:

.. code-block:: console

   $ export pybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")