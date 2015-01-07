import netCDF4
import numpy as np

import matplotlib as mpl
mpl.use('WXAgg')
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import basemap

emissivity = '94'

filename = 'data/dewpot/dew' + emissivity + 'ri_ysm.nc'

nc = netCDF4.Dataset(filename, 'r')
dew = nc.variables['sum'][:]  # kg/m2/d
lats_1d = nc.variables['lat'][:]
lons_1d = nc.variables['lon'][:]
lons, lats = np.meshgrid(lons_1d, lats_1d)

labels = ['DJF', 'MAM', 'JJA', 'SON']
vmin = 0.
vmax = 0.331
cmap = cm.RdYlBu
# cmap.set_over('#313695')
cmap.set_over('#00FF00')

my_dpi = 96

fig = plt.figure(figsize=(1200/my_dpi, 800./my_dpi))
for season in range(4):
    ax = fig.add_subplot(2, 2, season+1)
    map1 = basemap.Basemap(resolution='c', projection='kav7', lon_0=0)
    map1.drawcoastlines()
    # map1.drawcountries()
    map1.drawparallels(np.arange(-90, 91, 30))
    map1.drawmeridians(np.arange(0, 360, 60))

    dewpc = map1.pcolormesh(lons, lats, dew[season, :, :], latlon=True,
                            vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(labels[season])
    # ax.set_title(labels[season])

fig.tight_layout(pad=1, w_pad=1, h_pad=4)
# ax = fig.add_axes([0.11, 0.51, 0.8, 0.025])
ax = fig.add_axes([0.05, 0.52, 0.9, 0.025])
cb = plt.colorbar(cax=ax, orientation='horizontal', cmap=cmap,
                  extend='max', format="%.2f",
                  ticks=[0, 0.1, 0.2, 0.3, 0.4])

# plt.suptitle(title, fontsize=16)

plt.show()
