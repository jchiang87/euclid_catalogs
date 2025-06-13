"""
skyCatalogs interface to Euclid star catalogs.
"""
import os
from collections.abc import Iterable
import numpy as np
import pandas as pd
from astropy.io import fits
import galsim
from skycatalogs.objects import BaseObject, ObjectCollection
from .utils import MilkyWayExtinction, load_lsst_bandpasses, object_type_config


__all__ = ["EuclidStarCollection", "EuclidStar"]


MW_EXT = MilkyWayExtinction()
LSST_BPS = load_lsst_bandpasses()


class EuclidStar(BaseObject):

    def __init__(self, ra, dec, sed_index, Av, ref_mag,  # noqa: N803
                 obj_id, parent_collection, index):
        super().__init__(ra, dec, obj_id, parent_collection._object_type,
                         parent_collection, index)
        self.sed_index = sed_index
        self.Av = Av
        self.ref_mag = ref_mag
        self.collection = parent_collection

    def get_observer_sed_component(self, component, mjd=None):
        if component != "this_object":
            raise RuntimeError("Unknown SED component: %s", component)
        sed = self.collection.get_observed_sed(
            self.sed_index, self.Av, self.ref_mag)
        return sed

    def get_gsobject_components(self, gsparams=None, rng=None):
        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)
        return {'this_object': galsim.DeltaFunction(gsparams=gsparams)}


class EuclidStarCollection(ObjectCollection):
    _object_type = "Euclid_stars"

    def __init__(self, region, sky_catalog, sed_file, star_catalog_dir,
                 ref_band='r'):
        self._read_sed_data(sed_file)
        self.ref_band = ref_band
        self._read_catalog_files(region, star_catalog_dir)
        self._sky_catalog = sky_catalog
        self._object_type_unique = self._object_type
        self._object_class = EuclidStar
        self._uniform_object_type = True

    def _read_sed_data(self, sed_file):
        with fits.open(sed_file) as hdus:
            self.wl = hdus[1].data.copy()  # wavelengths in nm
            fnu = hdus[2].data.copy()  # fnu array in erg/cm^2/s/Hz
        # Convert to flambda units.  These spectra will be normalized
        # using a standard bandpass, so no need to convert the units
        # in detail.
        self.flambda = np.zeros(fnu.shape)
        for i in range(len(fnu)):
            self.flambda[i] = fnu[i]/self.wl**2

    def _read_catalog_files(self, region, star_catalog_dir):
        self._ra = []
        self._dec = []
        self._sed_index = []
        self._Av = []
        self._ref_mag = []
        self._id = []
        # Read in the index file, and do a crude selection of relevant
        # catalog files based on the bounds in ra, dec.
        index_file = os.path.join(star_catalog_dir,
                                  "star_catalog_index.parquet")
        df0 = pd.read_parquet(index_file)
        ra_min, ra_max, dec_min, dec_max = region.get_radec_bounds()

        selection = (f"ra_min < {ra_max} and {ra_min} < ra_max and "
                     f"dec_min < {dec_max} and {dec_min} < dec_max")
        df = df0.query(selection)
        for item in df['star_cat_file']:
            star_cat_file = os.path.join(star_catalog_dir, item)
            self._apply_region_mask(region, star_cat_file)

    def _apply_region_mask(self, region, star_cat_file):
        with fits.open(star_cat_file) as hdus:
            data = hdus[1].data.copy()
        ra = data['RA']
        dec = data['DEC']
        sed_index = data['SED_TEMPLATE']
        Av = data['AV']  # noqa: N806
        fnu = data[f'TU_FNU_{self.ref_band.upper()}_LSST']  # flux in jy
        obj_id = data['SOURCE_ID']
        mask = region.compute_mask(ra, dec)
        # Remove objects with non-positive fluxes
        mask = np.logical_or(mask, fnu <= 0.0)
        self._ra.extend(np.ma.array(ra, mask=mask).compressed())
        self._dec.extend(np.ma.array(dec, mask=mask).compressed())
        self._sed_index.extend(np.ma.array(sed_index, mask=mask).compressed())
        self._Av.extend(np.ma.array(Av, mask=mask).compressed())
        fnu = np.ma.array(fnu, mask=mask).compressed()
        ref_mag = -2.5*np.log10(fnu) + 8.9  # convert to AB mag
        self._ref_mag.extend(ref_mag)
        self._id.extend([str(_) for _ in
                         np.ma.array(obj_id, mask=mask).compressed()])

    def get_observed_sed(self, sed_index, Av, ref_mag):  # noqa: N803
        lut = galsim.LookupTable(self.wl, self.flambda[sed_index],
                                 interpolant='linear')
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')
        sed = MW_EXT.apply(sed, Av)
        sed = sed.withMagnitude(ref_mag, LSST_BPS[self.ref_band])
        return sed

    @property
    def native_columns(self):
        return ()

    def __getitem__(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            ra = self._ra[index]
            dec = self._dec[index]
            sed_index = self._sed_index[index]
            Av = self._Av[index]  # noqa: N806
            ref_mag = self._ref_mag[index]
            obj_id = self._id[index]
            return EuclidStar(ra, dec, sed_index, Av, ref_mag,
                              obj_id, self, index)
        elif isinstance(index, slice):
            return [self.__getitem__(i) for i in range(len(self))[index]]
        elif isinstance(index, tuple) and isinstance(index[0], Iterable):
            return [self.__getitem__(i) for i in index[0]]

    def __len__(self):
        return len(self._ra)

    @staticmethod
    def register(sky_catalog, object_type):
        sky_catalog.cat_cxt.register_source_type(
            EuclidStarCollection._object_type,
            object_class=EuclidStar,
            collection_class=EuclidStarCollection,
            custom_load=True
        )

    @staticmethod
    def load_collection(region, sky_catalog, mjd=None, **kwds):
        object_type = EuclidStarCollection._object_type
        config = object_type_config(sky_catalog, object_type)
        sed_file = config['sed_file']
        star_catalog_dir = config['star_catalog_dir']
        ref_band = config['ref_band']
        return EuclidStarCollection(region, sky_catalog, sed_file,
                                    star_catalog_dir, ref_band=ref_band)
