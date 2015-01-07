#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=False

cimport cython
import numpy as np
cimport numpy as np
import condenser as cr
cimport condenser as cr

# some functions from C libraries
cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)
    double log(double x)


# faster max
cdef inline double cmax(double a, double b) nogil:
    if a > b:
        return a
    else:
        return b


# calculate instantaneous values (>=0) from cumulative values
cpdef double[:] cumulative2instant(double[:] data, double resolution,
                                  int cumulated_steps):
    cdef int ntimes = data.shape[0]
    cdef int ii, jj, kk
    ii = 0
    cdef double[:] data_new = np.empty(ntimes, dtype=np.float64)
    cdef double prev

    # check if the cumulation start from 0
    prev = data[0]
    missed_steps = 0
    for jj in range(1, cumulated_steps):
        if (data[jj] < prev):
            missed_steps = cumulated_steps - jj
            break
        prev = data[jj]

    kk = missed_steps  # start counting from here

    while ii < ntimes:
        prev = 0.0
        for jj in range(kk, cumulated_steps):
            data_new[ii] = cmax(0., (data[ii] - prev) / ((kk+1) * resolution))  # >=0
            prev = data[ii]
            kk = 0
            ii += 1
            if ii >= ntimes:
                break
    return data_new


# estimate the wind speed at 2 m from data at 10 m
cpdef inline double[:] windspeed_to_2m(double[:] windspeed10, double[:] roughness):
    cdef int ii
    cdef int nn = windspeed10.shape[0]
    cdef double[:] windspeed2 = np.empty(nn)
    for ii in range(nn):
#        windspeed2[ii] = windspeed10[ii] * log(2./roughness[ii]) / log(10./roughness[ii])
        windspeed2[ii] = windspeed10[ii] * log((2.+roughness[ii])/roughness[ii]) \
                                         / log((10.+roughness[ii])/roughness[ii])
    return windspeed2


# interpolate data to some time instant between time0[0] .. time0[1]
cdef inline double interpdata(double time, double[:] time0,
                              double[:] data0):
    return data0[0] + ((data0[1] - data0[0]) / (time0[1] - time0[0])
                       * (time - time0[0]))


cdef inline double vapor_pressure_arm(double temperature):
    """
    Vapor pressure of water [Pa] according to the
    August-Roche-Magnus formula.
    Input: temperature [K]
    """
    return 610.94 * exp(17.625 * (temperature - 273.15)
                        / (temperature - 30.11))


# calculates the vapor pressure for all time instants
cdef inline double[:] vapor_pressure_forall(double[:] dewpoints):
    cdef int ii
    cdef int nn = dewpoints.shape[0]
    cdef double[:] vapor_pressures = np.empty(nn)
    for ii in range(nn):
        vapor_pressures[ii] = vapor_pressure_arm(dewpoints[ii])
    return vapor_pressures

# This interface function prepares input for the Condenser class and runs the dew model.
def get_dew(double[:] times, double[:] swcum, double[:] lwcum,
            double[:] winds10, double[:] roughness, double[:] temp,
            double[:] dewpoint, double timestep, double emissivity,
            double albedo, double heat_capacity, double mass,
            double input_resolution=3.*3600, int cumulated_steps=4,
            double output_timestep=3.*3600., convert_wind=True, rads_cumulative=True,
            cr.Heat_Transfer heat_transfer=cr.Heat_Transfer()):

    cdef int jj, day
    cdef double time1
    cdef int ntimes = temp.shape[0]
    cdef int nlons = temp.shape[1]
    cdef int nlats = temp.shape[2]
    cdef int stepsres = int(input_resolution/timestep)
    cdef double timestepfrac = timestep / (24. * 3600.)
    cdef double output_timestepfrac = output_timestep / (24. * 3600.)

    cdef double[:] vapor_pressure = np.empty_like(times)
    cdef double[:] swrad = np.empty_like(times)
    cdef double[:] lwrad = np.empty_like(times)
    cdef double[:] winds = np.empty_like(times)

    if convert_wind:
        winds = windspeed_to_2m(winds10, roughness)
    else:
        winds = winds10

    vapor_pressure = vapor_pressure_forall(dewpoint[:])

    if rads_cumulative:
        swrad = cumulative2instant(swcum[:], input_resolution, cumulated_steps)
        lwrad = cumulative2instant(lwcum[:], input_resolution, cumulated_steps)
    else:
        swrad = swcum
        lwrad = lwcum

    return np.asarray(cr.solve_water_RK4(emissivity, albedo, mass, heat_capacity, times,
                                         temp, vapor_pressure, winds, swrad, lwrad,
                                         timestepfrac, ntimes, stepsres,
                                         output_timestepfrac, heat_transfer))
