# Builtins
import logging

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize

# Locals
from oggm import entity_task
from oggm import utils, cfg
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance,
                                   RandomMassBalance)

from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist, clip_min, clip_max, clip_array)

from oggm.core.massbalance import MassBalanceModel, ConstantMassBalance
from oggm.core.flowline import FileModel, flowline_model_run
from oggm.exceptions import InvalidParamsError

# Constants
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR
from oggm.cfg import G, GAUSSIAN_KERNEL

# Module logger
log = logging.getLogger(__name__)

cluster_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/misc/magicc/scaling/'
DT_PER_DT_FILE = cluster_url + 'cmip_dt_per_dt_small.nc'
DT_PER_DT_MON_FILE = cluster_url + 'cmip_dt_per_dt_monthly_small.nc'
DP_PER_DT_FILE = cluster_url + 'cmip_dp_per_dt_small.nc'
DP_PER_DT_MON_FILE = cluster_url + 'cmip_dp_per_dt_monthly_small.nc'


@entity_task(log)
def parse_dt_per_dt(gdir, monthly=False):
    """Local climate change signal for this glacier added to the diagnostics
    """

    # Use xarray to read the data
    lon = gdir.cenlon + 360 if gdir.cenlon < 0 else gdir.cenlon
    lat = gdir.cenlat

    if monthly:
        convert = list
        tf = DT_PER_DT_MON_FILE
        tp = DP_PER_DT_MON_FILE
    else:
        convert = float
        tf = DT_PER_DT_FILE
        tp = DP_PER_DT_FILE

    with xr.open_dataset(utils.file_downloader(tf)) as ds:
        assert ds.longitude.min() >= 0
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        gdir.add_to_diagnostics('magicc_dt_per_dt', convert(ds['cmip_avg'].data))
    with xr.open_dataset(utils.file_downloader(tp)) as ds:
        assert ds.longitude.min() >= 0
        ds = ds.sel(longitude=lon, latitude=lat, method='nearest')
        gdir.add_to_diagnostics('magicc_dp_per_dt', convert(ds['cmip_avg'].data))


class MagiccConstantMassBalance(ConstantMassBalance):
    """Just overrides the call to interpolate which is useless here."""

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):
        _, m = floatyear_to_date(year)
        mb_on_h = heights*0.
        for yr in self.years:
            yr = date_to_floatyear(yr, m)
            mb_on_h += self.mbmod.get_monthly_mb(heights, year=yr)
        mb_on_h /= len(self.years)
        if add_climate:
            t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
            return mb_on_h, t, tmelt, prcp, prcpsol
        return mb_on_h

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        mb = heights*0.
        for yr in self.years:
            mb += self.mbmod.get_annual_mb(heights, year=yr)
        mb /= len(self.years)
        if add_climate:
            t, tmelt, prcp, prcpsol = self.get_annual_climate(heights)
            return mb, t, tmelt, prcp, prcpsol
        return mb


class MagiccMassBalance(MassBalanceModel):
    """Time-dependant Temp Bias ConstantMassBalance model
    """

    def __init__(self, gdir, magicc_ts=None, dt_per_dt=1, dp_per_dt=0,
                 mu_star=None, bias=None,
                 y0=None, halfsize=15, filename='climate_historical',
                 input_filesuffix='', **kwargs):
        """Initialize

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        magicc_ts : pd.Series
            the GMT time series
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the annual bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        dt_per_dt : float, optional, default 1
            the local climate change signal, in units of °C per °C
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        if magicc_ts is None:
            raise InvalidParamsError('Need a magicc ts!')

        super(MagiccMassBalance, self).__init__()
        self.mbmod = MagiccConstantMassBalance(gdir, mu_star=mu_star, bias=bias,
                                               y0=y0, halfsize=halfsize,
                                               filename=filename,
                                               input_filesuffix=input_filesuffix,
                                               **kwargs)

        self.valid_bounds = self.mbmod.valid_bounds
        self.hemisphere = gdir.hemisphere

        # Set ys and ye
        self.ys = int(magicc_ts.index[0])
        self.ye = int(magicc_ts.index[-1])

        # Correct for dp_per_dt signal
        if len(np.atleast_1d(dp_per_dt)) == 12:
            ref_t = magicc_ts.loc[y0-halfsize:y0+halfsize].mean()
            prcp_ts = (magicc_ts - ref_t).values[:, np.newaxis] * dp_per_dt
            prcp_ts = pd.DataFrame(data=prcp_ts, index=magicc_ts.index, columns=np.arange(1, 13))
        else:
            ref_t = magicc_ts.loc[y0-halfsize:y0+halfsize].mean()
            prcp_ts = (magicc_ts - ref_t) * dp_per_dt

        # We correct the original factor - don't forget to also scale the diff
        self.prcp_fac_ts = self.mbmod.prcp_fac + self.mbmod.prcp_fac * prcp_ts

        # Correct for dt_per_dt signal
        if len(np.atleast_1d(dt_per_dt)) == 12:
            magicc_ts = pd.DataFrame(data=magicc_ts.values[:, np.newaxis] * dt_per_dt,
                                     index=magicc_ts.index, columns=np.arange(1, 13))
        else:
            magicc_ts = magicc_ts * dt_per_dt

        years = magicc_ts.loc[y0-halfsize:y0+halfsize].index.values

        # OK now check the bias to apply based on y0 and halfsize
        fls = gdir.read_pickle('model_flowlines')
        mb_ref = PastMassBalance(gdir)
        mb_ref = mb_ref.get_specific_mb(fls=fls, year=years).mean()

        def to_minimize(temp_bias):
            self.temp_bias_ts = magicc_ts - temp_bias
            mb_mine = self.get_specific_mb(fls=fls, year=years).mean()
            return mb_mine - mb_ref

        temp_bias = optimize.brentq(to_minimize, -10, 10, xtol=1e-5)
        self.temp_bias_ts = magicc_ts - temp_bias

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def _check_bias(self, year):
        t = np.asarray(self.temp_bias_ts.loc[int(year)])
        if np.any(t != self.temp_bias):
            self.temp_bias = t
        p = np.asarray(self.prcp_fac_ts.loc[int(year)])
        if np.any(p != self.prcp_fac):
            self.prcp_fac = p

    def get_monthly_mb(self, heights, year=None, **kwargs):
        self._check_bias(year)
        return self.mbmod.get_monthly_mb(heights, year=year, **kwargs)

    def get_annual_mb(self, heights, year=None, **kwargs):
        self._check_bias(year)
        return self.mbmod.get_annual_mb(heights, year=year, **kwargs)


@entity_task(log)
def run_from_magicc_data(gdir, magicc_ts=None,
                         ys=None, ye=None,
                         y0=2014, halfsize=5,
                         use_dt_per_dt=True,
                         use_dp_per_dt=True,
                         climate_filename='climate_historical',
                         climate_input_filesuffix='', output_filesuffix='',
                         init_model_filesuffix=None, init_model_yr=None,
                         bias=None, **kwargs):
    """Runs a glacier with climate input from MAGICC.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr)
    ye : int
        end year of the model run (default: last year of the magicc file)
    y0 : int
        central year where to apply the MAGICC anomaly method
    halfsize : int
        half-size of the MAGICC anomaly window
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be
        combined with `init_model_yr`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the y0 - halfsize
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    if init_model_yr is None:
        init_model_yr = y0 - halfsize

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_geometry', filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)
        if init_model_yr is None:
            init_model_yr = fmod.last_yr
        # Avoid issues here
        if init_model_yr > fmod.y0:
            fmod.run_until(init_model_yr)
        else:
            fmod.run_until(fmod.y0)

        init_model_fls = fmod.fls
        if ys is None:
            ys = init_model_yr
    else:
        raise InvalidParamsError('We strongly recommend to start from '
                                 'prepro for this task.')

    # Take from rgi date if not set yet
    if ys is None:
        raise InvalidParamsError('ys should not be guessed at this point')

    dt_per_dt = 1.
    dp_per_dt = 0.
    if use_dt_per_dt:
        dt_per_dt = np.asarray(gdir.get_diagnostics()['magicc_dt_per_dt'])
    if use_dp_per_dt:
        dp_per_dt = np.asarray(gdir.get_diagnostics()['magicc_dp_per_dt'])

    # Final crop
    mb = MagiccMassBalance(gdir, magicc_ts=magicc_ts, y0=y0, halfsize=halfsize,
                           dt_per_dt=dt_per_dt, dp_per_dt=dp_per_dt,
                           filename=climate_filename, bias=bias,
                           input_filesuffix=climate_input_filesuffix)

    if ye is None:
        # Decide from climate
        ye = mb.ye

    return flowline_model_run(gdir, output_filesuffix=output_filesuffix,
                              mb_model=mb, ys=ys, ye=ye,
                              init_model_fls=init_model_fls,
                              **kwargs)
