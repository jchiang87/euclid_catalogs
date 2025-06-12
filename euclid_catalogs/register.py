from .utils import object_type_config
from .stars import EuclidStarCollection


__all__ = ['register_objects']


COLLECTION_CLASS_MAP = {"EuclidStarCollection": EuclidStarCollection}


def register_objects(sky_catalog, object_type):
    config = object_type_config(sky_catalog, object_type)
    if "collection_class" not in config:
        return
    collection_class = COLLECTION_CLASS_MAP[config["collection_class"]]
    collection_class.register(sky_catalog, object_type)
