import numpy as np
import netCDF4

from dew_interface import get_dew
import heat_transfer_coeffs as htc


from jug.task import TaskGenerator


np.seterr(all='raise')


# set global vars
ifile = netCDF4.Dataset('data/temp.nc', 'r')
times = ifile.variables['time'][:].astype(np.float64)  # hours since beginning of dataset
lats = ifile.variables['lat'][:]  # 90..-90
lons = ifile.variables['lon'][:]  # 0..360
ntimes, nlats, nlons = ifile.variables['temp'].shape
ifile.close()

timestep = 10.  # s
sheet_mass = 1. * 0.00039 * 920.
emissivity = 0.94
albedo = 0.84
heat_capacity = 2300.


# sets a task to run the dew model for one grid point
@TaskGenerator
def get_one(ilat, ilon):
    # first load input data for current grid point
    ifile = netCDF4.Dataset('data/temp.nc', 'r')
    temp = ifile.variables['temp'][:, ilat, ilon].astype(np.float64)
    ifile.close()
    ifile = netCDF4.Dataset('data/shortwave.nc', 'r')
    swcum = ifile.variables['shortwave'][:, ilat, ilon].astype(np.float64)
    ifile.close()
    ifile = netCDF4.Dataset('data/longwave.nc', 'r')
    lwcum = ifile.variables['longwave'][:, ilat, ilon].astype(np.float64)
    ifile.close()
    ifile = netCDF4.Dataset('data/winds.nc', 'r')
    winds10 = ifile.variables['winds'][:, ilat, ilon].astype(np.float64)
    ifile.close()
    ifile = netCDF4.Dataset('data/dewpoint.nc', 'r')
    dewpoint = ifile.variables['dewpoint'][:, ilat, ilon].astype(np.float64)
    ifile.close()
    ifile = netCDF4.Dataset('data/srough.nc', 'r')
    srough = ifile.variables['var244'][:, ilat, ilon].astype(np.float64)
    ifile.close()

    lon = lons[ilon]
    if lon > 180.:
        lon -= 360.  # convert western longitudes to 0..-180
    local_datenum = (times + int(lon/15)) / 24.  # convert to local time in ~datenum format

    return ilat, ilon, get_dew(local_datenum, swcum, lwcum,
                               winds10, srough, temp, dewpoint, timestep,
                               emissivity, albedo, heat_capacity, sheet_mass,
                               heat_transfer=htc.Beysens2005(),
                               output_timestep=24.*3600.)[:, ::2]


# sets a single task for writing output
@TaskGenerator
def writeout(args):
    # count the number of different calendar days
    ndays = np.unique(np.floor(times/24.)).shape[0]
    # create and setup a NetCDF file for output
    ofile = netCDF4.Dataset('dewpot/dew_out.nc', 'w')
    ofile.createDimension('lat', nlats)
    ofile.createDimension('lon', nlons)
    ofile.createDimension('time', ndays)
    olats = ofile.createVariable('lat', 'f4', ('lat',))
    olons = ofile.createVariable('lon', 'f4', ('lon',))
    otime = ofile.createVariable('time', 'f8', ('time',))
    odew = ofile.createVariable('dew', 'f4', ('time', 'lat', 'lon',),
                                chunksizes=(ndays, 1, 1))
    oice = ofile.createVariable('ice', 'f4', ('time', 'lat', 'lon',),
                                chunksizes=(ndays, 1, 1))

    ofile.description = 'Modeled dew collection'
    olats.long_name = 'latitude'
    olats.units = 'degree_north'
    olats.standard_name = 'latitude'
    olats.axis = 'Y'
    olons.long_name = 'longitude'
    olons.units = 'degree_east'
    olons.standard_name = 'longitude'
    olons.axis = 'X'
    otime.long_name = 'time'
    otime.units = 'days since 1979-01-01 00:00:00.0'
    otime.standard_name = 'time'
    otime.calendar = 'proleptic_gregorian'
    otime.axis = 'T'
    odew.units = 'kg/m2/d'
    oice.units = 'kg/m2/d'

    # these are the same as in input
    olats[:] = lats
    olons[:] = lons
    # otime[:] = times[1:]  # ok, we lose the first 3 hours
    otime[:] = np.arange(ndays)

    # go through all the model outputs
    for ilat, ilon, data in args:
        odew[:, ilat, ilon] = data[:, 0]
        oice[:, ilat, ilon] = data[:, 1]
    ofile.close()


# create tasks to run the dew model for each grid point + 1 for output
writeout([get_one(ilat, ilon)
          for ilat in range(nlats)
          for ilon in range(nlons)])
