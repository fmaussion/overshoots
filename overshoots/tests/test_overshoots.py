import warnings
warnings.filterwarnings("once", category=DeprecationWarning)  # noqa: E402

import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
import numpy as np
import xarray as xr
import pandas as pd

# Local imports
import oggm
from oggm.core import massbalance, climate
from oggm import cfg, utils, workflow, tasks
from oggm.cfg import SEC_IN_DAY, SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import get_demo_file
from oggm.core.flowline import init_present_time_glacier

from overshoots.massbalance import MagiccMassBalance, run_from_magicc_data, parse_dt_per_dt

prepro_base_url = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/'
                   'L3-L5_files/ERA5/elev_bands/qc3/pcp1.6/match_geod/')
magicc_dir = 'https://cluster.klima.uni-bremen.de/~fmaussion/misc/magicc/magicc_ensemble_sel/'

DO_PLOT = True

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
        cfg.PARAMS['use_multiprocessing'] = True

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

        if DO_PLOT:
            plt.figure()
            df[['mb_ref', 'mb', 'mb_nodt']].loc[2000:2025].plot(title=rid);
            plt.figure()
            df[['mb_ref', 'mb', 'mb_nodt']].loc[2000:].plot(title=rid);
            plt.show()

    def test_varying_bias_workflow(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir
        cfg.PARAMS['use_multiprocessing'] = True

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

    def test_hydro_workflow(self, case_dir):

        cfg.initialize()
        cfg.PARAMS['prcp_scaling_factor'] = 1.6
        cfg.PATHS['working_dir'] = case_dir
        cfg.PARAMS['use_multiprocessing'] = True
        cfg.PARAMS['store_diagnostic_variables'] = ['volume', 'area']

        # Go - get the pre-processed glacier directories
        rgi_ids = ['RGI60-14.06794', 'RGI60-11.00897']
        gdirs = workflow.init_glacier_directories(rgi_ids,
                                                  from_prepro_level=5,
                                                  prepro_base_url=prepro_base_url,
                                                  prepro_border=80,
                                                  prepro_rgi_version='62')
        workflow.execute_entity_task(parse_dt_per_dt, gdirs)

        exp = 'netzero_py2050_fac1.0_decr0.3'
        magicc_file = magicc_dir + exp + '.nc'
        with xr.open_dataset(utils.file_downloader(magicc_file), decode_times=False) as ds:
            ds = ds.load()
        df = ds['ens_avg'].to_dataframe()

        workflow.execute_entity_task(tasks.run_with_hydro, gdirs,
                                     run_task=run_from_magicc_data,
                                     magicc_ts=df['ens_avg'],
                                     init_model_filesuffix='_historical',
                                     output_filesuffix='_' + exp)

        ds = utils.compile_run_output(gdirs, filesuffix='_' + exp)

        for rgi_id in rgi_ids:
            odf = ds.sel(rgi_id=rgi_id).to_dataframe().iloc[:-1]

            # Sanity checks
            # Tot prcp here is constant (constant climate)
            odf['tot_prcp'] = (odf['liq_prcp_off_glacier'] +
                               odf['liq_prcp_on_glacier'] +
                               odf['snowfall_off_glacier'] +
                               odf['snowfall_on_glacier'])
            assert_allclose(odf['tot_prcp'], odf['tot_prcp'].iloc[0])

            # Glacier area is the same (remove on_area?)
            assert_allclose(odf['on_area'], odf['area'])

            # Our MB is the same as the glacier dyn one
            reconstructed_vol = (odf['model_mb'].cumsum() / cfg.PARAMS['ice_density'] +
                                 odf['volume'].iloc[0])
            assert_allclose(odf['volume'].iloc[1:], reconstructed_vol.iloc[:-1],
                            atol=1e-2)

            # Mass-conservation
            odf['runoff'] = (odf['melt_on_glacier'] +
                             odf['melt_off_glacier'] +
                             odf['liq_prcp_on_glacier'] +
                             odf['liq_prcp_off_glacier'])

            mass_in_glacier_end = odf['volume'].iloc[-1] * cfg.PARAMS['ice_density']
            mass_in_glacier_start = odf['volume'].iloc[0] * cfg.PARAMS['ice_density']

            mass_in_snow = odf['snow_bucket'].iloc[-1]
            mass_in = odf['tot_prcp'].iloc[:-1].sum()
            mass_out = odf['runoff'].iloc[:-1].sum()
            assert_allclose(mass_in_glacier_end,
                            mass_in_glacier_start + mass_in - mass_out - mass_in_snow,
                            atol=1e-2)  # 0.01 kg is OK as numerical error

            # Qualitative assessments
            assert odf['melt_on_glacier'].iloc[-1] < odf['melt_on_glacier'].iloc[0] * 0.7
            assert odf['liq_prcp_off_glacier'].iloc[-1] > odf['liq_prcp_on_glacier'].iloc[-1]
            assert odf['liq_prcp_off_glacier'].iloc[0] < odf['liq_prcp_on_glacier'].iloc[0]

            # Residual MB should not be crazy large
            frac = odf['residual_mb'] / odf['runoff']
            assert_allclose(frac, 0, atol=0.1)

            if DO_PLOT:
                plt.figure()
                odf[['volume']].plot(title=rgi_id)

                plt.figure()
                odf[['melt_on_glacier',
                     'melt_off_glacier',
                     'liq_prcp_on_glacier',
                     'liq_prcp_off_glacier']].plot.area(title=rgi_id)

                plt.figure()
                frac.plot(title=rgi_id)

                plt.show()
