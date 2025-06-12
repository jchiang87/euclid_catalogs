import os
import numpy as np
from astropy import units as u
from dust_extinction.parameter_averages import F19
import galsim


__all__ = ["MilkyWayExtinction", "load_lsst_bandpasses",
           "object_type_config"]


class MilkyWayExtinction:
    """
    Class to apply Milky Way extinction to an SED using the
    Fitzpatrick, et al. (2019) model from the dust_extinction package.
    """
    def __init__(self, Rv=3.1, delta_wl=1.0, eps=1e-7):
        wl_min = 1e3/F19.x_range[1] + eps
        wl_max = 1e3/F19.x_range[0] - eps
        npts = int((wl_max - wl_min)/delta_wl)
        self.wl = np.linspace(wl_min, wl_max, npts)
        self.extinction = F19(Rv=Rv)

    def apply(self, sed, Av):
        ext = self.extinction.extinguish(self.wl*u.nm, Av=Av)
        lut = galsim.LookupTable(self.wl, ext, interpolant='linear')
        mw_ext = galsim.SED(lut, wave_type='nm', flux_type='1').thin()
        sed = sed*mw_ext
        return sed


def load_lsst_bandpasses():
    """Load the LSST bandpasses from the rubin_sim distribution."""
    bandpass_dir = os.path.join(os.environ["RUBIN_SIM_DATA_DIR"],
                                "throughputs", "baseline")
    lsst_bandpasses = {}
    bands = "ugrizy"
    for band in bands:
        bp_file = os.path.join(bandpass_dir, f"total_{band}.dat")
        bp = galsim.Bandpass(bp_file, "nm")
        bp = bp.truncate(relative_throughput=1e-3)
        bp = bp.thin()
        bp = bp.withZeropoint("AB")
        lsst_bandpasses[band] = bp
    return lsst_bandpasses


def object_type_config(sky_catalog, object_type):
    return dict(sky_catalog.raw_config["object_types"][object_type])
