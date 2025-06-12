import os
import glob
from astropy.io import fits
import numpy as np
from euclid_catalogs.utils import FieldRotator

galaxy_catalog_dir = "/home/jchiang/work/Rubin/Euclid-Rubin_DDP_IWG/Euclid_catalogs/Galcat_fs2_dr1_stage1"

pattern = os.path.join(galaxy_catalog_dir, "a*", "EUC-GALCAT_STANDARD*.fits")
galaxy_cat_files = sorted([_ for _ in glob.glob(pattern)
                           if "_rotated" not in _])

# Reference center to use for the original galaxies.
ra0 = 262.8
dec0 = 63.5

# New desired reference center for rotated positions.
ra1 = 96.0
dec1 = -43.0

for galaxy_catalog in galaxy_cat_files:
    print(os.path.basename(galaxy_catalog))
    with fits.open(galaxy_catalog) as hdus:
        ra = hdus[1].data['RA']
        dec = hdus[1].data['DEC']


        rotator = FieldRotator(ra0, dec0, ra1, dec1)

        ra_new, dec_new = rotator.transform(ra, dec)

        hdus[1].data['RA'] = ra_new
        hdus[1].data['DEC'] = dec_new
        outfile = galaxy_catalog.replace(".fits", "_rotated.fits")
        hdus.writeto(outfile)
