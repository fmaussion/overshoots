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

from overshoots.massbalance import MagiccMassBalance, run_from_magicc_data, parse_dt_per_dt

prepro_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/match_geod/'
magicc_dir = 'https://cluster.klima.uni-bremen.de/~fmaussion/misc/magicc/magicc_ensemble_sel/'
magicc_meta = 'https://cluster.klima.uni-bremen.de/~fmaussion/misc/magicc/magicc_ensemble_meta.csv'


class TestMassBalanceModels:

    def test_dt_per_dt(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897', 'RGI60-03.04384'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=80,
                                                  prepro_rgi_version='62')

        workflow.execute_entity_task(parse_dt_per_dt, gdirs)

        dt1 = gdirs[0].get_diagnostics()['magicc_dt_per_dt']
        dt2 = gdirs[1].get_diagnostics()['magicc_dt_per_dt']
        assert dt1 < 0.7 * dt2

    def test_varying_bias_mb_model(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir

        # Go - get the pre-processed glacier directories
        for rid in ['RGI60-03.04384', 'RGI60-11.00897', 'RGI60-16.02207']:
            gdirs = workflow.init_glacier_directories([rid],
                                                      from_prepro_level=5,
                                                      prepro_base_url=prepro_base_url,
                                                      prepro_border=80,
                                                      prepro_rgi_version='62')
            workflow.execute_entity_task(parse_dt_per_dt, gdirs)

            gdir = gdirs[0]

            exp = 'netzero_py2020_fac1.0_decr0.3'
            magicc_file = magicc_dir + exp + '.nc'
            with xr.open_dataset(utils.file_downloader(magicc_file), decode_times=False) as ds:
                ds = ds.load()
            df = ds['ens_avg'].to_dataframe()
            dt_per_dt = gdir.get_diagnostics()['magicc_dt_per_dt']

            fls = gdir.read_pickle('model_flowlines')

            y0 = 2014
            hs = 5

            mbc = massbalance.ConstantMassBalance(gdir, y0=y0, halfsize=hs)
            mb_ref = massbalance.PastMassBalance(gdir)
            years = np.arange(1980, 2020, dtype=np.float64)
            df['mb_ref'] = pd.Series(index=years,
                                     data=mb_ref.get_specific_mb(fls=fls, year=years))

            mb = MagiccMassBalance(gdir, y0=y0, halfsize=hs,
                                   magicc_ts=df['ens_avg'],
                                   dt_per_dt=dt_per_dt)
            df['mb'] = mb.get_specific_mb(fls=fls, year=df.index)

            mb = MagiccMassBalance(gdir, y0=y0, halfsize=hs,
                                   magicc_ts=df['ens_avg'])
            df['mb_nodt'] = mb.get_specific_mb(fls=fls, year=df.index)

            assert_allclose(df.loc[y0-hs:y0+hs]['mb'].mean(),
                            mbc.get_specific_mb(fls=fls),
                            rtol=5e-3)
            assert_allclose(df[['ens_avg', 'mb']].corr().min().min(), -1, rtol=1e-2)
            assert_allclose(df[['ens_avg', 'mb_nodt']].corr().min().min(), -1, rtol=1e-2)

        #     import matplotlib.pyplot as plt
        #     plt.figure()
        #     df[['mb_ref', 'mb', 'mb_nodt']].loc[2000:2025].plot(title=rid);
        #     plt.figure()
        #     df[['mb_ref', 'mb', 'mb_nodt']].loc[2000:].plot(title=rid);
        #
        # plt.show()

    def test_varying_bias_workflow(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir

        # Go - get the pre-processed glacier directories
        gdirs = workflow.init_glacier_directories(['RGI60-11.00897'],
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=80,
                                                  prepro_rgi_version='62')
        workflow.execute_entity_task(parse_dt_per_dt, gdirs)
        gdir = gdirs[0]

        exp = 'netzero_py2020_fac1.0_decr0.3'
        magicc_file = magicc_dir + exp + '.nc'
        with xr.open_dataset(utils.file_downloader(magicc_file), decode_times=False) as ds:
            ds = ds.load()
        df = ds['ens_avg'].to_dataframe()

        run_from_magicc_data(gdir, magicc_ts=df['ens_avg'],
                             init_model_filesuffix='_historical',
                             output_filesuffix='_'+exp)

        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix='_'+exp)) as ds:
            df['vol'] = ds.volume_m3.to_series()
            assert ds.time[0] == 2020
            assert ds.time[-1] == 2301
            assert ds.volume_m3.isel(time=0) > ds.volume_m3.isel(time=-1)
            assert ds.volume_m3.min() < ds.volume_m3.isel(time=-1)
