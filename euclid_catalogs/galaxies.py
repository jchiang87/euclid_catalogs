import galsim
from skycatalogs.objects import BaseObject, ObjectCollection


__all__ = ["EuclidGalaxyCollection", "register_objects"]


def register_objects(sky_catalog):
    EuclidGalaxyCollection.register(sky_catalog)


class EuclidGalaxyObject(BaseObject):

    def __init__(self):
        super().__init__(self)

    def get_observer_sed_component(self, component, mjd=None):
        if component != "this_object":
            raise RuntimeError("Unknown SED component: %s", component)
        return self.sed

    def get_gsobject_components(self, gsparams=None, rng=None):
        if gsparams is not None:
            gsparams = galsim.GSParams(**gsparams)
        return {'this_object': galsim.DeltaFunction(gsparams=gsparams)}


class EuclidGalaxyCollection(ObjectCollection):
    _object_type = "Euclid_galaxies"
    def __init__(self, region, sky_catalog):
        pass

    @property
    def native_columns(self):
        return ()

    def __len__(self):
        return 0

    @staticmethod
    def register(sky_catalog):
        sky_catalog.cat_cxt\
            .register_source_type(EuclidGalaxyCollection._object_type,
                                  object_class=EuclidGalaxyObject,
                                  collection_class=EuclidGalaxyCollection,
                                  custom_load=True)

    @staticmethod
    def load_collection(region, sky_catalog, mjd=None, exposure=None):
        return EuclidGalaxyCollection(region, sky_catalog)
