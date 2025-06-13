"""
skyCatalogs interface to Euclid galaxy catalogs.
"""
import os
from collections import namedtuple
from collections.abc import Iterable
import numpy as np
import pandas as pd
from astropy.io import fits
import galsim
from skycatalogs.objects import BaseObject, ObjectCollection
from .utils import MilkyWayExtinction, load_lsst_bandpasses, object_type_config


__all__ = ["EuclidGalaxyCollection", "EuclidGalaxy"]


MW_EXT = MilkyWayExtinction()
LSST_BPS = load_lsst_bandpasses()


_GALAXY_FIELDS = ("SOURCE_ID", "RA", "DEC", "Z_OBS", "BULGE_R50",
                  "BULGE_NSERSIC", "BULGE_AXIS_RATIO", "INCLINATION_ANGLE",
                  "KAPPA", "GAMMA1", "GAMMA2", "SED_TEMPLATE_1",
                  "EXT_LAW_1", "EBV_1", "AV", "REF_MAG")

EuclidGalaxyParams = namedtuple("EuclidGalaxyParams", _GALAXY_FIELDS)


class EuclidGalaxy(BaseObject):

    def __init__(self, params, parent_collection, index):
        super().__init__(params.RA,
                         params.DEC,
                         params.SOURCE_ID,
                         parent_collection._object_type,
                         parent_collection,
                         index)
        self.params = params
        self.collection = parent_collection

    def get_observer_sed_component(self, component, mjd=None):
        if component not in self.subcomponents:
            raise RuntimeError("Unknown SED component: %s", component)
        sed = self.collection.get_observed_sed(component, self.params)
        return sed

    def get_gsobject_components(self, gsparams=None, rng=None):
        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)
        obj_dict = {}
        for component in self.subcomponents:
            obj_dict[component] = self.get_gsobject(component, self.params,
                                                    gsparams)
        return obj_dict

    def get_gsobject(self, component, galaxy_params, gsparams):
        if component != "bulge":
            raise RuntimeError("Only modeling galaxy bulge components")
        # Create the azimuthally symmetric sersic object.
        obj = galsim.Sersic(n=galaxy_params.BULGE_NSERSIC,
                            half_light_radius=galaxy_params.BULGE_R50,
                            gsparams=gsparams)
        # Apply shear to get the desired shape and inclination.
        eta = -np.log(galaxy_params.BULGE_AXIS_RATIO)
        beta = galaxy_params.INCLINATION_ANGLE*galsim.degrees
        shear = galsim.Shear(eta=eta, beta=beta)
        obj = obj._shear(shear)
        # Apply weak lensing parameters.
        g1 = galaxy_params.GAMMA1/(1. - galaxy_params.KAPPA)
        g2 = galaxy_params.GAMMA2/(1. - galaxy_params.KAPPA)
        mu = 1./((1. - galaxy_params.KAPPA)**2
                 - (galaxy_params.GAMMA1**2 + galaxy_params.GAMMA2**2))
        obj = obj._lens(g1, g2, mu)
        return obj


class EuclidGalaxyCollection(ObjectCollection):
    _object_type = "Euclid_galaxies"

    def __init__(self, region, sky_catalog, sed_file, ext_law_file,
                 galaxy_catalog_dir, ref_band='r'):
        self._read_sed_data(sed_file)
        self._read_ext_law_data(ext_law_file)
        self.ref_band = ref_band
        self._ref_fnu_column = f"TU_FNU_{ref_band.upper()}_LSST_MAG"
        self._read_catalog_files(region, galaxy_catalog_dir)
        self._sky_catalog = sky_catalog
        self._object_type_unique = self._object_type
        self._object_class = EuclidGalaxy
        self._uniform_object_type = True

    @property
    def subcomponents(self):
        # Restrict to modeling just the bulge component (as a sersic)
        # since it's unclear how the disk component should be modeled.
        return ["bulge"]

    def _read_sed_data(self, sed_file):
        with fits.open(sed_file) as hdus:
            self.wl = hdus[1].data.copy()  # wavelengths in nm
            self.flambda = hdus[2].data.copy()  # flambda in erg/s/cm2/Angstrom

    def _read_ext_law_data(self, ext_law_file):
        pass

    def _read_catalog_files(self, region, galaxy_catalog_dir):
        self._ra = []
        self._dec = []
        self._id = []
        self.params = []
        index_file = os.path.join(galaxy_catalog_dir,
                                  "galaxy_catalog_index.parquet")
        df0 = pd.read_parquet(index_file)
        ra_min, ra_max, dec_min, dec_max = region.get_radec_bounds()
        selection = (f"ra_min < {ra_max} and {ra_min} < ra_max and "
                     f"dec_min < {dec_max} and {dec_min} < dec_max")
        df = df0.query(selection)
        for item in df['galaxy_cat_file']:
            galaxy_cat_file = os.path.join(galaxy_catalog_dir, item)
            self._apply_region_mask(region, galaxy_cat_file)

    def _apply_region_mask(self, region, galaxy_cat_file):
        with fits.open(galaxy_cat_file)as hdus:
            data = hdus[1].data.copy()
        ra = data['RA']
        dec = data['DEC']
        fnu = data[self._ref_fnu_column]
        mask = region.compute_mask(ra, dec)
        mask = np.logical_or(mask, fnu <= 0.0)
        param_columns = {}
        for column in _GALAXY_FIELDS[:-1]:
            param_columns[column] \
                = np.ma.array(data[column], mask=mask).compressed()
        param_columns['SOURCE_ID'] = [str(_) for _ in
                                      param_columns['SOURCE_ID']]
        fnu = np.ma.array(fnu, mask=mask).compressed()
        param_columns['REF_MAG'] = -2.5*np.log10(fnu) + 8.9  # fnu -> AB mag
        self._ra.extend(param_columns['RA'])
        self._dec.extend(param_columns['DEC'])
        self._id.extend(param_columns['SOURCE_ID'])
        df = pd.DataFrame(param_columns)
        self.params.extend([EuclidGalaxyParams(*(row.to_list()))
                            for _, row in df.iterrows()])

    def get_observed_sed(self, component, params):
        sed_index = int(params.SED_TEMPLATE_1)
        lut = galsim.LookupTable(self.wl, self.flambda[sed_index],
                                 interpolant='linear')
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
        # Neglect intrinsic reddening and just apply Milky Way extinction.
        sed = MW_EXT.apply(sed, params.AV)
        sed = sed.withMagnitude(params.REF_MAG, LSST_BPS[self.ref_band])
        sed = sed.atRedshift(params.Z_OBS)
        return sed

    @property
    def native_columns(self):
        return ()

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            return EuclidGalaxy(self.params[index], self, index)
        elif isinstance(index, slice):
            return [self.__getitem__(i) for i in range(len(self))[index]]
        elif isinstance(index, tuple) and isinstance(index[0], Iterable):
            return [self.__getitem__(i) for i in index[0]]

    def __len__(self):
        return len(self._ra)

    @staticmethod
    def register(sky_catalog, object_type):
        sky_catalog.cat_cxt.register_source_type(
            EuclidGalaxyCollection._object_type,
            object_class=EuclidGalaxy,
            collection_class=EuclidGalaxyCollection,
            custom_load=True
        )

    @staticmethod
    def load_collection(region, sky_catalog, **kwds):
        object_type = EuclidGalaxyCollection._object_type
        config = object_type_config(sky_catalog, object_type)
        sed_file = config['sed_file']
        ext_law_file = config['ext_law_file']
        galaxy_catalog_dir = config['galaxy_catalog_dir']
        ref_band = config['ref_band']
        return EuclidGalaxyCollection(region, sky_catalog, sed_file,
                                      ext_law_file, galaxy_catalog_dir,
                                      ref_band=ref_band)
