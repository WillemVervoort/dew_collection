from condenser cimport Heat_Transfer

# A set of heat transfer coefficients that can be used with the Condenser class.

cdef extern from "math.h":
    double sqrt(double x)

cdef double CHAR_LENGTH = 1.

cdef class Richards2009(Heat_Transfer):
    label = "Richards (2009)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 5.9 + 4.1 * wind_speed * 805. / (511. + temperature_air)

cdef class Beysens2005(Heat_Transfer):
    label = "Beysens et al. (2005)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 4. * sqrt(wind_speed / CHAR_LENGTH)

cdef class Maestrevalero2012(Heat_Transfer):
    label = "Maestre-Valero et al. (2012)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 7.6 + 6.6 * wind_speed * (511. + 294.)/(511. + temperature_air)

cdef class Sparrow1979(Heat_Transfer):
    label = "Sparrow et al. (1979)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 4.96 * sqrt(wind_speed / CHAR_LENGTH)

cdef class Jurges1924(Heat_Transfer):
    label = "Jurges (1924)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 5.7 + 3.8 * wind_speed

cdef class Wattmuff1977(Heat_Transfer):
    label = "Wattmuff et al. (1977)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 2.8 + 3. * wind_speed

cdef class Test1981(Heat_Transfer):
    label = "Test et al. (1981)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 8.55 + 2.56 * wind_speed

cdef class Kumar1997(Heat_Transfer):
    label = "Kumar et al. (1997)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 10.03 + 4.687 * wind_speed

cdef class Lunde1980(Heat_Transfer):
    label = "Lunde (1980)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 4.5 + 2.9 * wind_speed

cdef class Sharples1998(Heat_Transfer):
    label = "Sharples and Charlesworth (1998)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15): 
        return 9.4 * sqrt(wind_speed)

cdef class Sharples1998a(Heat_Transfer):
    label = "Sharples and Charlesworth (1998)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 6.5 + 3.3 * wind_speed

cdef class Kumar2010(Heat_Transfer):
    label = "Kumar and Mullick (2010)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15):
        return 6.9 + 3.87 * wind_speed

cdef class Kumar2010a(Heat_Transfer):
    label = "Kumar and Mullick (2010)"
    cpdef double coeff(self, double wind_speed=1., double temperature_air=293.15): 
        return 6.63 + 3.87 * wind_speed**0.8 / CHAR_LENGTH**0.2
