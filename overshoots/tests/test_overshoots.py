import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

from numpy.testing import assert_allclose
import numpy as np
import xarray as xr
import pandas as pd

# Local imports
import oggm
from oggm.core import massbalance, climate
from oggm import cfg, utils, workflow
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import get_demo_file
from oggm.core.flowline import init_present_time_glacier

from overshoots.massbalance import MagiccMassBalance, run_from_magicc_data


base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/gdirs/final_prepro_cmip6/cera5_eb/'
magicc_file = 'https://cluster.klima.uni-bremen.de/~fmaussion/misc/magicc_os_ts.csv'


class TestMassBalanceModels:

    def test_varying_bias_mb_model(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.8
        cfg.PATHS['working_dir'] = case_dir

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=base_url,
                                                  prepro_border=80,
                                                  prepro_rgi_version='62')

        gdir = gdirs[0]

        df = pd.read_csv(utils.file_downloader(magicc_file), index_col=0)

        exp = 'netzero_py2020_fac1.0_decr0.3'
        mb = MagiccMassBalance(gdir, y0=2008, halfsize=10,
                               magicc_ts=df[exp])

        mbc = massbalance.ConstantMassBalance(gdir, y0=2008, halfsize=10)

        fls = gdir.read_pickle('model_flowlines')
        df['mb'] = mb.get_specific_mb(fls=fls, year=df.index)

        assert_allclose(df.loc[1998:2018]['mb'].mean(),
                        mbc.get_specific_mb(fls=fls),
                        rtol=1e-3)
        assert_allclose(df[[exp, 'mb']].corr().min().min(),
                        -1,
                        rtol=1e-3)

    def test_varying_bias_workflow(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.8
        cfg.PATHS['working_dir'] = case_dir

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=base_url,
                                                  prepro_border=80,
                                                  prepro_rgi_version='62')

        gdir = gdirs[0]

        df = pd.read_csv(utils.file_downloader(magicc_file), index_col=0)

        exp = 'netzero_py2020_fac1.0_decr0.3'

        run_from_magicc_data(gdir, magicc_ts=df[exp],
                             init_model_filesuffix='_historical',
                             output_filesuffix='_'+exp)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_'+exp)) as ds:
            df['vol'] = ds.volume_m3.to_series()
            assert ds.time[0] == 2019
            assert ds.time[-1] == 2301
            assert ds.volume_m3.isel(time=0) > ds.volume_m3.isel(time=-1)
            assert ds.volume_m3.min() < ds.volume_m3.isel(time=-1)
