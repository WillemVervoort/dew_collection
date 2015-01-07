import numpy as np
import netCDF4
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams.update({'font.size': 20, 'legend.fontsize': 20})

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mpldates

import heat_transfer_coeffs
from dew_interface import get_dew
from matplotlib.dates import date2num, num2date


np.seterr(all='raise')

ifile = netCDF4.Dataset('data/helsinki.nc', 'r')
timevar = ifile.variables['time']
all_times = netCDF4.num2date(timevar[:], timevar.units)

start_date = datetime(2000, 9, 2, 12, 0)
end_date = datetime(2000, 9, 5, 0, 0)
selinds = ((all_times >= start_date) & (all_times <= end_date))

time = all_times[selinds] + timedelta(0, 3*3600.)  # local time
days = (timevar[selinds].astype(np.float64) + 3.) / 24.  # local time
# lon = 24.0
# days = (timevar[selinds].astype(np.float64) + int(lon/15)) / 24.  # local time

ilat = ilon = 1
temp = ifile.variables['temp'][selinds, ilat, ilon].astype(np.float64)
swcum = ifile.variables['shortwave'][selinds, ilat, ilon].astype(np.float64)
lwcum = ifile.variables['longwave'][selinds, ilat, ilon].astype(np.float64)
winds = ifile.variables['winds'][selinds, ilat, ilon].astype(np.float64)
dewpoint = ifile.variables['dewpoint'][selinds, ilat, ilon].astype(np.float64)
ifile.close()
#srough = np.ones_like(winds)
srough = np.genfromtxt('data/srough_helsinki.txt')[selinds]


timestep = 10.  # s
sheet_mass = 1. * 0.00039 * 920.
emissivity = 0.94
albedo = 0.84
heat_capacity = 2300.


data = get_dew(days, swcum, lwcum, winds, srough, temp, dewpoint,
                    timestep, emissivity, albedo, heat_capacity, sheet_mass,
                    output_timestep=3*3600.,
                    heat_transfer=heat_transfer_coeffs.Richards2009())

dew = data[:, 0]
temp_c = data[:, 1]
ice = data[:, 2]

temp_c = np.where(temp_c < 100., np.nan, temp_c)

output_times = np.array([num2date(num)
                         for num in np.linspace(date2num(start_date)+1./8.,
                                                date2num(end_date)+1./8.,
                                                dew.shape[0])[:]])


# calculate instantaneous values (>=0) from cumulative values
def cumulative2instant(data, resolution, cumulated_steps, missed_steps):
    ntimes = data.shape[0]
    ii = 0
    kk = missed_steps  # start counting from here
    data_new = np.empty(ntimes)

    while ii < ntimes:
        prev = 0.0
        for jj in range(kk, cumulated_steps):
            data_new[ii] = max(0., (data[ii] - prev) / ((kk+1) * resolution))  # >=0
            prev = data[ii]
            kk = 0
            ii += 1
            if ii >= ntimes:
                break
    return data_new


# estimate the wind speed at 2 m from data at 10 m
def windspeed_to_2m(windspeed10, roughness):
    nn = windspeed10.shape[0]
    windspeed2 = np.empty(nn)
    for ii in range(nn):
        windspeed2[ii] = windspeed10[ii] * (
            np.log(2./roughness[ii]) / np.log(10./roughness[ii]))
    return windspeed2


winds2 = windspeed_to_2m(winds, srough)
swrad = cumulative2instant(swcum, 3.*3600., 4, 3)
lwrad = cumulative2instant(lwcum, 3.*3600., 4, 3)


plotind = 3
fig, (ax3, ax1) = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=.4)

# fig = plt.figure(figsize=(8,6))
# ax1 = fig.add_subplot(111)
pt = ax1.plot(time[plotind:-1], temp[plotind:-1]-273.15, '#1b9e77', linewidth=5, linestyle='-',
         marker='v', markersize=15, markeredgecolor='#1b9e77', alpha=0.9, label='Air temperature')
pdp = ax1.plot(time[plotind:-1], dewpoint[plotind:-1]-273.15, '#7570b3', linewidth=5, linestyle='-',
         marker='o', markersize=15, markeredgecolor='#7570b3', alpha=0.9, label='Dewpoint')
ptc = ax1.plot(output_times[1:-1], temp_c[:-2]-273.15, '#d95f02', linewidth=5, linestyle='-',
         alpha=0.9, label='Sheet temperature')
ax1.set_xlabel('Local time')
ax1.set_ylabel('Temperature [$^\circ$C]')
ax1.yaxis.set_major_locator(MaxNLocator(prune='lower'))
ax1.set_ylim([0, 17])
ax1.set_yticks([3, 6, 9, 12, 15])

ax2 = ax1.twinx()
ax2.bar(output_times-timedelta(0, 0.*3600.), dew, align='center',
        color='lightblue', edgecolor='none', width=0.1)
ax2.set_ylabel('Cumulative dew [mm]')
ax2.set_ylim([0., 0.43])
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

ax2.plot([time[plotind+4], time[plotind+4]], [0, 1], '--k')
ax2.plot([time[plotind+12], time[plotind+12]], [0, 1], '--k')

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')


# Figure 2

psw = ax3.plot(time[plotind:-1], swrad[plotind:-1], '#e6ab02', linewidth=5,
               linestyle='-', marker='o', markersize=15, markeredgecolor='#e6ab02',
               alpha=0.9, label='Shortwave radiation')
# psw[0].set_clip_on(False)
plw = ax3.plot(time[plotind:-1], lwrad[plotind:-1], '#66a61e', linewidth=5,
               linestyle='-', alpha=0.9, label='Longwave radiation')
ax3.set_ylabel('Radiation [W/m$^2$]')
ax3.set_xticklabels([])
ax3.patch.set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.xaxis.tick_top()
ax3.set_ylim([-15, 400])
ax3.set_yticks([0, 100, 200, 300, 400])

ax4 = ax3.twinx()
pws = ax4.plot(time[plotind:-1], winds2[plotind:-1], '#e7298a', 
               linewidth=5, linestyle='-', marker='v', markersize=15,
               markeredgecolor='#e7298a', alpha=0.9, label='Wind speed')
ax4.set_ylabel('Wind speed [m/s]')
ax4.set_ylim([0, 4.5])
ax4.set_yticks([0, 1, 2, 3, 4])
# ax4.yaxis.label.set_color(pws.get_color())

all_plots = psw + pt + plw + pdp + pws + ptc
all_labels = [p.get_label() for p in all_plots]
leg = ax1.legend(all_plots, all_labels, ncol=3, columnspacing=1)
leg.draggable(True)
leg.set_frame_on(False)

ax2.set_xlim([time[plotind], time[-2]])
fig.autofmt_xdate()
ax1.xaxis.set_major_formatter(mpldates.DateFormatter('%H:%M'))

plt.show()
