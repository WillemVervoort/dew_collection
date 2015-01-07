#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np
cimport numpy as cnp

cdef double SPEC_HEAT_AIR = 1.01E3  # [J/kg/K]
cdef double LATENT_FUSION = 334000.0  # [J/kg]
cdef double LATENT_VAPORIZATION = 2.26E6  # [J/kg]
cdef double LATENT_SUBLIMATION = LATENT_VAPORIZATION + LATENT_FUSION
cdef double HEAT_CAPACITY_WATER = 4181.3  # [J/kg/K]
cdef double HEAT_CAPACITY_ICE = 2110.  # [J/kg/K]
cdef double MOLEC_WEIGHT_RATIO = 0.622  # [1], water vapor / dry air


# some functions from C libraries
cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)


# faster max
cdef inline double cmax(double a, double b) nogil:
    if a > b:
        return a
    else:
        return b

# faster abs
cdef inline double cabs(double a) nogil:
    if a < 0.:
        return -a
    else:
        return a


# interpolate data to some time instant between time0[0] .. time0[1] (timeterm below)
cdef inline double interp(double timeterm, double[:] data0) nogil:
    return data0[0] + (data0[1] - data0[0]) * timeterm

# the precalculated timeterm used in interpolation
cdef inline double interp_timeterm(double time, double[:] time0) nogil:
    return (time - time0[0]) / (time0[1] - time0[0])


# returns the index of first larger -1
cdef inline int find_first(double data, double[:] dataset) nogil:
    cdef int low = 0
    cdef int high = dataset.shape[0]-1
    cdef int ii = (high - low) / 2
    while (dataset[ii] > data) or (dataset[ii+1] <= data):
        ii = low + (high - low) / 2
        if dataset[ii] > data:
            high = ii
        else:
            low = ii
    return ii


cdef inline double black_body(double temperature) nogil:
    """
    Black body radiation according to Stefan-Boltzmann's law [W/m^2].
    Input: temperature [K].
    """
    return 5.6704E-8 * temperature**4.


cdef inline double vapor_pressure_arm(double temperature):
    """
    Vapor pressure of water [Pa] according to the
    August-Roche-Magnus formula.
    Input: temperature [K]
    """
    return 610.94 * exp(17.625 * (temperature - 273.15)
                        / (temperature - 30.11))


cdef inline double emiss_sky(double cloud_frac, double temperature) nogil:
    """
    Emissitivity of the sky [1].
    Input: cloud fraction (0..1), temperature [K].
    """
    if (cloud_frac == 0.):
        return 0.72 + 0.005 * (temperature - 273.15)
    cdef double emiss_sky0 = emiss_sky(0., temperature)
    return emiss_sky0 + cloud_frac * (1. - emiss_sky0 - 8. / temperature)


# Overridable class to allow different parameterizations efficiently
cdef class Heat_Transfer:
    # defaults to Richards, 2009
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 5.9 + 4.1 * wind_speed * 805. / (511. + temperature_air)


# the class that includes all parameters and the state of the condenser sheet
cdef class Condenser:
    cdef double latitude
    cdef double longitude
    cdef double height
    cdef double emissivity
    cdef double absorptivity_sw
    cdef double heat_capacity
    cdef double area
    cdef double char_length_inv
    cdef double area_exch
    cdef double mass
    cdef double temperature
    cdef double water_mass
    cdef double ice_mass
    cdef double cond_rate
    cdef int daycounter
    cdef Heat_Transfer heat_transfer

    def __init__(self, double latitude=0, double longitude=0, double height=0,
                 double emissivity=0.94, double heat_capacity=1.9E3,
                 double area=1.0, double area_exch=1.0, double mass=1.0,
                 double temperature=293.15, double albedo=0.84,
                 Heat_Transfer heat_transfer=Heat_Transfer()):
        self.latitude = latitude
        self.longitude = longitude
        self.height = height
        self.emissivity = emissivity
        self.absorptivity_sw = 1. - albedo
        self.heat_capacity = heat_capacity
        self.area = area
        self.char_length_inv = 1. / sqrt(self.area)
        self.area_exch = area_exch
        self.mass = mass
        self.temperature = temperature
        self.water_mass = 0.
        self.ice_mass = 0.
        self.cond_rate = 0.
        self.daycounter = 1
        self.heat_transfer = heat_transfer

    # assume ratio between transfer coefficients equals the psychrometric constant
    cdef inline double mass_transfer_coeff(self, double wind_speed, double pressure=101300.,
                                           double temperature_air=293.15):
        return MOLEC_WEIGHT_RATIO * self.heat_transfer.coeff(wind_speed, temperature_air) \
               / (pressure * SPEC_HEAT_AIR)

    # incoming longwave irradiance to condenser
    cdef inline double rad_longwave_in(self, double temperature_air, double cloud_frac):
        return (emiss_sky(cloud_frac, temperature_air) * black_body(temperature_air))

    # outgoing irradiance from condenser
    cdef inline double rad_out(self):
        return black_body(self.temperature)

    # total heat from irradiance terms
    # NOTE: shortwave and longwave components assumed orthogonal to surface
    cdef inline double heat_rad(self, double shortwave_down, double longwave_down):
        return (shortwave_down * self.absorptivity_sw 
                + (longwave_down - self.rad_out()) * self.emissivity
                ) * self.area

    # convective heat exchange
    cdef inline double heat_conv_exch(self, double temperature_air, double wind_speed):
        return (self.area_exch * self.heat_transfer.coeff(wind_speed, temperature_air)
                * (temperature_air - self.temperature))

    # latent heat (rate) from condensation
    cdef inline double heat_cond(self):
        if self.temperature < 273.15:  # desublimation
            return LATENT_SUBLIMATION * self.cond_rate
        else:
            return LATENT_VAPORIZATION * self.cond_rate

    # condensation rate dm/dt
    cdef inline double cond_rate_eq(self, double temperature_air, double vapor_pressure,
                                    double wind_speed):
        return cmax(0., self.area_exch * self.mass_transfer_coeff(wind_speed)
                    * (vapor_pressure - vapor_pressure_arm(self.temperature)))

    # heat flow in
    cdef inline double heat_in(self, double shortwave_down, double longwave_down,
                      double temperature_air, double wind_speed):
        return (self.heat_rad(shortwave_down, longwave_down)
                + self.heat_conv_exch(temperature_air, wind_speed)
                + self.heat_cond())

    # rate of change of condensator temperature
    cdef inline double temp_rate_eq(self, heat_in):
        return heat_in / (self.mass * self.heat_capacity
                          + self.water_mass * HEAT_CAPACITY_WATER
                          + self.ice_mass * HEAT_CAPACITY_ICE)

    # rate of mass melting (when positive) or freezing (when negative)
    cdef inline double melt_rate(self, heat_in):
        return heat_in / LATENT_FUSION

    # returns the derivatives of water_mass and temperature (C version)
    cdef void _derivative(self, double[:] values, double temperature_air1,
                              vapor_pressure1, double wind_speed1,
                              shortwave_down1, double longwave_down1,
                              double[:] derivs):

        self.water_mass = values[0]
        self.temperature = values[1]
        self.ice_mass = values[2]
        self.cond_rate = self.cond_rate_eq(temperature_air1, vapor_pressure1, wind_speed1)

        # if sheet temperature is below freezing point, first freeze all water before allowing
        # the temperature to decrease further
        cdef double heat_in = self.heat_in(shortwave_down1, longwave_down1,
                                           temperature_air1, wind_speed1)*3600.*24. # [/s] -> [/d]
        if self.temperature < 273.15:
            derivs[2] = self.cond_rate*3600.*24.  # [/s] -> [/d]
            if (self.water_mass > 0.) & (heat_in < 0.):
                derivs[0] = self.melt_rate(heat_in)
                derivs[2] -= derivs[0]
                derivs[1] = 0.
            else:
                derivs[0] = 0.
                derivs[1] = self.temp_rate_eq(heat_in)

        else:
            derivs[0] = self.cond_rate*3600.*24.  # [/s] -> [/d]
            if (self.ice_mass > 0.) & (heat_in > 0.):
                derivs[2] = -self.melt_rate(heat_in)
                derivs[0] -= derivs[2]
                derivs[1] = 0.
            else:
                derivs[1] = self.temp_rate_eq(heat_in)
                derivs[2] = 0.
        return


    # integrates condensation rate and change in temperature with RK4 (Python compatible)
    def update_RK4(self, double[:] datenums, double[:] temperature_air,
                double[:] vapor_pressure, double[:] wind_speed,
                double[:] shortwave_down, double[:] longwave_down,
                double timestep, int ntimes, int stepsres,
                double output_timestep):
        return self._update_RK4(datenums, temperature_air,
                vapor_pressure, wind_speed, shortwave_down,
                longwave_down, timestep, ntimes, stepsres, output_timestep)

    # integrates condensation rate and change in temperature with RK4
    cdef double[:, :] _update_RK4(self, double[:] datenums,
                double[:] temperature_air, double[:] vapor_pressure,
                double[:] wind_speed, double[:] shortwave_down,
                double[:] longwave_down, double timestep, int ntimes,
                int stepsres, double output_timestep):

        cdef int output_times = int((datenums[ntimes-1]-datenums[0]) / output_timestep) + 1
        cdef int output_step = int(output_timestep / timestep)
        cdef int reset_step = int(1. / timestep)
        cdef double[:, :] output = np.zeros((output_times, 3))
        cdef double[:] values = np.empty(3)
        cdef double[:] derivs = np.zeros(3, dtype=np.float64)

        cdef double[:] k1 = np.empty(3)
        cdef double[:] k2 = np.empty(3)
        cdef double[:] k3 = np.empty(3)
        cdef double[:] k4 = np.empty(3)

        cdef double water_mass0
        cdef double temperature0
        cdef double ice_mass0

        cdef double water_mass_max = 0.
        cdef double ice_mass_max = 0.

        cdef int ii, jj, output_index, output_counter, inow, reset_counter
        cdef double timeterm, temperature_air1, vapor_pressure1, wind_speed1
        cdef double shortwave_down1, longwave_down1
        cdef double time

        # shift time by 12 hours so that daynumber changes at noon (for reset)
        for ii in range(ntimes):
            datenums[ii] += 0.5

        output_index = 0
        output_counter = int((datenums[0] % output_timestep) / timestep)  # fix first write if t[0] != 0
        reset_counter = int(datenums[0] % 1.0 / timestep)  # offset for daily reset

        # loop over all inputs
        for ii in range(0, ntimes-1):
            time = datenums[ii]  
            for jj in range(stepsres):  # loop over timesteps per input

                if (reset_counter == reset_step):  # reset daily yield at even datenum
                    # print "Reset at ", time
                    self.water_mass = 0.
                    self.ice_mass = 0.
                    reset_counter = 0
                    
                inow = ii
                water_mass0 = self.water_mass
                temperature0 = self.temperature
                ice_mass0 = self.ice_mass

                # timestep so small that these assumed constant during one RK4 step
                timeterm = interp_timeterm(time, datenums[inow:inow+2])
                temperature_air1 = interp(timeterm, temperature_air[inow:inow+2])
                vapor_pressure1 = interp(timeterm, vapor_pressure[inow:inow+2])
                wind_speed1 = interp(timeterm, wind_speed[inow:inow+2])
                shortwave_down1 = interp(timeterm, shortwave_down[inow:inow+2])
                longwave_down1 = interp(timeterm, longwave_down[inow:inow+2])

                #Runge-Kutta 4 solver
                values[0] = water_mass0
                values[1] = temperature0
                values[2] = ice_mass0
                self._derivative(values, temperature_air1,
                                      vapor_pressure1, wind_speed1, shortwave_down1,
                                      longwave_down1, k1)

                values[0] = water_mass0 + k1[0] * timestep / 2.
                values[1] = temperature0 + k1[1] * timestep / 2.
                values[2] = ice_mass0 + k1[2] * timestep / 2.
                self._derivative(values, temperature_air1,
                                      vapor_pressure1, wind_speed1, shortwave_down1,
                                      longwave_down1, k2)

                values[0] = water_mass0 + k2[0] * timestep / 2.
                values[1] = temperature0 + k2[1] * timestep / 2.
                values[2] = ice_mass0 + k2[2] * timestep / 2.
                self._derivative(values, temperature_air1,
                                      vapor_pressure1, wind_speed1, shortwave_down1,
                                      longwave_down1, k3)

                values[0] = water_mass0 + k3[0] * timestep
                values[1] = temperature0 + k3[1] * timestep
                values[2] = ice_mass0 + k3[2] * timestep
                self._derivative(values, temperature_air1,
                                      vapor_pressure1, wind_speed1, shortwave_down1,
                                      longwave_down1, k4)

                self.water_mass = water_mass0 + (k1[0] + 2. * k2[0] + 2. * k3[0]
                                                 + k4[0]) * timestep / 6.
                self.temperature = temperature0 + (k1[1] + 2. * k2[1] + 2. * k3[1]
                                                 + k4[1]) * timestep / 6.
                self.ice_mass = ice_mass0 + (k1[2] + 2. * k2[2] + 2. * k3[2]
                                                 + k4[2]) * timestep / 6.

                # a rude correction for "overmelting/overfreezing"
                if self.water_mass < 0.:
                    self.ice_mass += self.water_mass
                    self.water_mass = 0.
                if self.ice_mass < 0.:
                    self.water_mass += self.ice_mass
                    self.ice_mass = 0.

                # store max total water mass
                if self.water_mass + self.ice_mass > water_mass_max + ice_mass_max:
                    water_mass_max = self.water_mass
                    ice_mass_max = self.ice_mass

                time += timestep
                reset_counter += 1
                output_counter += 1

                # write output
                if output_counter % output_step == 0:
                    # print output_index, time, time%output_timestep, ntimes, output_times, output_step
                    output[output_index, 0] = water_mass_max
                    output[output_index, 1] = self.temperature
                    output[output_index, 2] = ice_mass_max
                    water_mass_max = 0.
                    ice_mass_max = 0.
                    output_index += 1

        return output


# The function to call from Python code. Solves everything in one go.
def solve_water_RK4(double emissivity, double albedo, double mass, 
                double heat_capacity, double[:] datenums, 
                double[:] temperature_air,
                double[:] vapor_pressure, double[:] wind_speed,
                double[:] shortwave_down, double[:] longwave_down,
                double timestep, int ntimes, int stepsres,
                double output_timestep, Heat_Transfer heat_transfer=Heat_Transfer()):

    c = Condenser(emissivity=emissivity, albedo=albedo, mass=mass,
                  heat_capacity=heat_capacity, heat_transfer=heat_transfer)
    return c._update_RK4(datenums, temperature_air, vapor_pressure,
                         wind_speed, shortwave_down, longwave_down, 
                         timestep, ntimes, stepsres, output_timestep)
