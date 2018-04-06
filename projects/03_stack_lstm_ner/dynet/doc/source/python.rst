Installing DyNet for Python
===========================

Python bindings to DyNet are supported for both Python 2.x and 3.x.
Before installing DyNet, you will need to make sure that several packages are installed.
For example on **Ubuntu Linux**:

::
    
    sudo apt-get update
    sudo apt-get install python-pip build-essential cmake mercurial

Or on **macOS**, first make sure the Apple Command Line Tools are installed, then
get CMake, and Mercurial with either homebrew or macports:

::

    xcode-select --install
    brew install cmake hg python # Using homebrew.
    sudo port install cmake mercurial py-pip # Using macports.

On **Windows**, see :ref:`windows-python-install`.


Once these packages are installed, the following will download, build and install
DyNet. Note that compiling DyNet may take a long time, up to 10 minutes or more, but as
long as you see "Running setup.py install for dynet" with the moving progress
wheel, things should be running.

.. code:: bash

    pip install git+https://github.com/clab/dynet#egg=dynet

Alternatively, you can add the following to your `requirements.txt`:

.. code:: bash

    git+https://github.com/clab/dynet#egg=dynet

In case installation using `pip` fails, if you copy-and-paste the entire log that you
get after running the `pip` command into a `github issue <https://github.com/clab/dynet/issues>`_,
we will help you debug. You can also try to install DyNet manually as listed below.

Manual Installation
-------------------

The following is a list of all the commands needed to perform a manual install:

.. code:: bash

    # Installing Python DyNet:

    pip install cython  # if you don't have it already.
    mkdir dynet-base
    cd dynet-base
    # getting dynet and eigen
    git clone https://github.com/clab/dynet.git
    hg clone https://bitbucket.org/eigen/eigen -r 346ecdb  # -r NUM specified a known working revision
    cd dynet
    mkdir build
    cd build
    # without GPU support (if you get an error that Eigen cannot be found, try using the full path to Eigen)
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python`
    # or with GPU support (if you get an error that Eigen cannot be found, try using the full path to Eigen)
    cmake .. -DEIGEN3_INCLUDE_DIR=../../eigen -DPYTHON=`which python` -DBACKEND=cuda

    make -j 2 # replace 2 with the number of available cores
    cd python
    python setup.py install  # or `python setup.py install --user` for a user-local install.
    
    # this should suffice, but on some systems you may need to add the following line to your
    # init files in order for the compiled .so files be accessible to Python.
    # /path/to/dynet/build/dynet is the location in which libdynet.dylib resides.
    export DYLD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$DYLD_LIBRARY_PATH


To explain these one-by-one, first we get DyNet:

.. code:: bash

    cd $HOME
    mkdir dynet-base
    cd dynet-base
    git clone https://github.com/clab/dynet.git
    cd dynet
    git submodule init # To be consistent with DyNet's installation instructions.
    git submodule update # To be consistent with DyNet's installation instructions.

Then get Eigen:

.. code:: bash

    cd $HOME
    cd dynet-base
    hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb
    
(`-r NUM` specifies a known working revision of Eigen. You can remove this in order to get the bleeding
edge Eigen, with the risk of some compile breaks, and the possible benefit of added optimizations.)

We also need to make sure the ``cython`` module is installed. (you can
replace ``pip`` with your favorite package manager, such as ``conda``,
or install within a virtual environment)

.. code:: bash

    pip install cython

To simplify the following steps, we can set a bash variable to hold
where we have saved the main directories of DyNet and Eigen. In case you
have gotten DyNet and Eigen differently from the instructions above and
saved them in different location(s), these variables will be helpful:

.. code:: bash

    PATH_TO_DYNET=$HOME/dynet-base/dynet/
    PATH_TO_EIGEN=$HOME/dynet-base/eigen/

Compile DyNet.

This is pretty much the same process as compiling DyNet, with the
addition of the ``-DPYTHON=`` flag, pointing to the location of your
Python interpreter.

Assuming that the ``cmake`` command found all the needed libraries and
didn't fail, the ``make`` command will take a while, and compile DyNet
as well as the Python bindings. You can change ``make -j 2`` to a higher
number, depending on the available cores you want to use while
compiling.

You now have a working Python binding inside of ``build/dynet``. To
verify this is working:

.. code:: bash

    cd $PATH_TO_DYNET/build/python
    python

then, within Python:

.. code:: bash

    import dynet as dy
    print dy.__version__
    pc = dy.ParameterCollection()

In order to install the module so that it is accessible from everywhere
in the system, run the following:

.. code:: bash

    cd $PATH_TO_DYNET/build/python
    python setup.py install --user

The ``--user`` switch will install the module in your local
site-packages, and works without root privileges. To install the module
to the system site-packages (for all users), or to the current `virtualenv`
(if you are on one), run ``python setup.py install`` without this switch.

You should now have a working python binding (the ``dynet`` module).

Note however that the installation relies on the compiled DyNet library
being in ``$PATH_TO_DYNET/build/dynet``, so make sure not to move it
from there.

Now, check that everything works:

.. code:: bash

    cd $PATH_TO_DYNET
    cd examples/python
    python xor.py
    python rnnlm.py rnnlm.py

Alternatively, if the following script works for you, then your
installation is likely to be working:

::

    from dynet import *
    pc = ParameterCollection()

If it doesn't work and you get an error similar to the following:
::

    ImportError: dlopen(/Users/sneharajana/.python-eggs/dyNET-0.0.0-py2.7-macosx-10.11-intel.egg-tmp/_dynet.so, 2): Library not loaded: @rpath/libdynet.dylib
    Referenced from: /Users/sneharajana/.python-eggs/dyNET-0.0.0-py2.7-macosx-10.11-intel.egg-tmp/_dynet.so
    Reason: image not found``

then you may need to run the following (and add it to your shell init files):

    export DYLD_LIBRARY_PATH=/path/to/dynet/build/dynet/:$DYLD_LIBRARY_PATH

# /path/to/dynet/build/dynet is the location in which libdynet.dylib resides.

Anaconda Support
----------------

`Anaconda 
<https://www.continuum.io/downloads>`_ is a popular package management system for Python, and DyNet can be installed into this environment.
First, make sure that you install all the necessary packages according to the instructions at the top of this page.
Then create an Anaconda environment and activate it as below:

::

     source activate my_environment_name

After this, you should be able to install using pip or manual installation as normal.

.. _windows-python-install:

Windows Support
---------------

You can also use Python on Windows. For simplicity, we recommend 
using a Python distribution that already has Cython installed. The following has been tested to work:

1) Install WinPython 2.7.10 (comes with Cython already installed).
2) Compile DyNet according to the directions in the Windows C++ documentation (:ref:`windows-cpp-install`), and additionally add the following flag when executing ``cmake``: ``-DPYTHON=/path/to/your/python.exe``.
3) Open a command prompt and set ``VS90COMNTOOLS`` to the path to your Visual Studio "Common7/Tools" directory. One easy way to do this is a command such as:

::

    set VS90COMNTOOLS=%VS140COMNTOOLS%

4) Open dynet.sln from this command prompt and build the "Release" version of the solution.
5) Follow the rest of the instructions above for testing the build and installing it for other users

Note, currently only the Release version works.

GPU/MKL Support
---------------

Installing on GPU
~~~~~~~~~~~~~~~~~

For installing on a computer with GPU, first install CUDA. The following
instructions assume CUDA is installed.

The installation process is pretty much the same, while adding the
``-DBACKEND=cuda`` flag to the ``cmake`` stage:

.. code:: bash

    cmake .. -DEIGEN3_INCLUDE_DIR=$PATH_TO_EIGEN -DPYTHON=$PATH_TO_PYTHON -DBACKEND=cuda

(if CUDA is installed in a non-standard location and ``cmake`` cannot
find it, you can specify also
``-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda``.)

Now, build the Python modules (as above, we assume Cython is installed):

After running ``make -j 2``, you should have the files ``_dynet.so`` and
``_gdynet.so`` in the ``build/python`` folder.

As before, ``cd build/python`` followed by
``python setup.py install --user`` will install the module.



Using the GPU from Python
~~~~~~~~~~~~~~~~~~~~~~~~~

The preferred way to make dynet use the GPU under Python is to import
dynet as usual:

::

    import dynet

Then tell it to use the GPU by using the commandline switch
``--dynet-gpu`` or the GPU switches detailed `here
<commandline.html>`__ when invoking the program. This option lets the
same code work with either the GPU or the CPU version depending on how
it is invoked.

Alternatively, you can also select whether the CPU or GPU should be
used by using one of the following more specific import statements:

::

    import _dynet
    # or
    import _gdynet # For GPU

This may be useful if you want to decide programmatically whether to
use the CPU or GPU. Importantly, importing ``_dynet`` or ``_gdynet``
will not initialize the global parameters. If you forget to initialize
these, dynet may abort with a segmentation fault. Instead, make sure
to initialize the global parameters, as follows:

::

    # Same as import dynet as dy
    import _dynet as dy
    dy.init()


Running with MKL
~~~~~~~~~~~~~~~~

If you've built DyNet to use MKL (using ``-DMKL`` or ``-DMKL_ROOT``), Python sometimes has difficulty finding
the MKL shared libraries. You can try setting ``LD_LIBRARY_PATH`` to point to your MKL library directory.
If that doesn't work, try setting the following environment variable (supposing, for example,
your MKL libraries are located at ``/opt/intel/mkl/lib/intel64``):

.. code:: bash

    export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so


