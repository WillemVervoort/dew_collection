cdef class Heat_Transfer:
    cpdef double coeff(self, double wind_speed=*, double temperature_air=*)
