# A model to estimate dew collection on artificial surfaces

#### Needs meteorological input such as ECMWF's ERA-Interim:

- shortwave radiation
- longwave radiation
- horizontal wind speed
- ambient temperature
- dew point

#### A brief description of the included files:

File | Description
:---:| ---
condenser.pxd | Cython declarations for condenser.pyx.    
condenser.pyx | The Condenser class that includes the dew model.
daily_dew_jug.py | A global application of the model using [Jug](https://github.com/luispedro/jug).
dew_interface.pyx | The interface for the dew model. Take a look at dew_interface.get_dew().
example_2days.py | An example of using the model as a fast box model + plotting.
heat_transfer_coeffs.pyx | Implementations of some heat transfer coefficients.
Makefile | A very simple Makefile.
plot_dewpot.py | An example of plotting global model output.
README.md | This.
run_global.sh | A SLURM shell script for running on cluster.
setup.py | The Makefile uses this to build the Cython files.
