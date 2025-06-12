import os
import numpy as np
from astropy import units as u
from dust_extinction.parameter_averages import F19
import galsim
from rubin_scheduler.utils import (cartesian_from_spherical,
                                   spherical_from_cartesian)
from rubin_scheduler.utils import (rotation_matrix_from_vectors,
                                   angular_separation)


__all__ = ["MilkyWayExtinction", "load_lsst_bandpasses",
           "object_type_config", "FieldRotator"]


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


class FieldRotator(object):

    def __init__(self, ra0, dec0, ra1, dec1):
        """
        The source of the code in this class can be found here:

        https://github.com/LSSTDESC/sims_GCRCatSimInterface/blob/master/python/desc/sims/GCRCatSimInterface/ProtoDC2DatabaseEmulator.py#L41

        Parameters
        ----------
        ra0, dec0 are the coordinates of the original field
        center in degrees

        ra1, dec1 are the coordinates of the new field center
        in degrees

        The transform() method of this class operates by first
        applying a rotation that carries the original field center
        into the new field center.  Points are then transformed into
        a basis in which the unit vector defining the new field center
        is the x-axis.  A rotation about the x-axis is applied so that
        a point that was due north of the original field center is still
        due north of the field center at the new location.  Finally,
        points are transformed back into the original x,y,z bases.
        """

        # do we actually need to do the rotation, or is the simulation
        # already in the right spot?
        self._needs_to_be_rotated = True
        rot_dist = angular_separation(ra0, dec0, ra1, dec1)
        if rot_dist < 1.0/3600.0:
            self._needs_to_be_rotated = False
            return

        # find the rotation that carries the original field center
        # to the new field center
        xyz = cartesian_from_spherical(np.radians(ra0), np.radians(dec0))
        xyz1 = cartesian_from_spherical(np.radians(ra1), np.radians(dec1))
        if np.abs(1.0-np.dot(xyz, xyz1)) < 1.0e-10:
            self._transformation = np.identity(3, dtype=float)
            return

        first_rotation = rotation_matrix_from_vectors(xyz, xyz1)

        # create a basis set in which the unit vector
        # defining the new field center is the x axis
        xx = np.dot(first_rotation, xyz)
        rng = np.random.RandomState(99)
        mag = np.nan
        while np.abs(mag) < 1.0e-20 or np.isnan(mag):
            random_vec = rng.random_sample(3)
            comp = np.dot(random_vec, xx)
            yy = random_vec - comp*xx
            mag = np.sqrt((yy**2).sum())
            yy /= mag

        zz = np.cross(xx, yy)

        to_self_bases = np.array([xx,
                                  yy,
                                  zz])

        out_of_self_bases = to_self_bases.transpose()

        # Take a point due north of the original field
        # center.  Apply first_rotation to carry it to
        # the new field.  Transform it to the [xx, yy, zz]
        # bases and find the rotation about xx that will
        # make it due north of the new field center.
        # Finally, transform back to the original bases.
        d_dec = 0.1
        north = cartesian_from_spherical(np.radians(ra0),
                                         np.radians(dec0+d_dec))

        north = np.dot(first_rotation, north)

        # print(np.degrees(sphericalFromCartesian(north)))

        north_true = cartesian_from_spherical(np.radians(ra1),
                                              np.radians(dec1+d_dec))

        north = np.dot(to_self_bases, north)
        north_true = np.dot(to_self_bases, north_true)
        north = np.array([north[1], north[2]])
        north /= np.sqrt((north**2).sum())
        north_true = np.array([north_true[1], north_true[2]])
        north_true /= np.sqrt((north_true**2).sum())

        c = north_true[0]*north[0]+north_true[1]*north[1]
        s = north[0]*north_true[1]-north[1]*north_true[0]
        norm = np.sqrt(c*c+s*s)
        c = c/norm
        s = s/norm

        yz_rotation = np.array([[1.0, 0.0, 0.0],
                                [0.0, c, -s],
                                [0.0, s, c]])

        second_rotation = np.dot(out_of_self_bases,
                                 np.dot(yz_rotation,
                                        to_self_bases))

        self._transformation = np.dot(second_rotation,
                                      first_rotation)

    def transform(self, ra, dec):
        """
        ra, dec are in degrees; return the RA, Dec coordinates
        of the point about the new field center
        """
        xyz = cartesian_from_spherical(np.radians(ra),
                                       np.radians(dec)).transpose()
        xyz = np.dot(self._transformation, xyz).transpose()
        ra_out, dec_out = spherical_from_cartesian(xyz)
        return np.degrees(ra_out), np.degrees(dec_out)
