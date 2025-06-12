import os
import glob
from collections import defaultdict
from astropy.io import fits
import numpy as np
import pandas as pd

star_cat_files = sorted(glob.glob('a*/EUC_SIM_STAR-CATALOG*.fits'))

data = defaultdict(list)
for item in star_cat_files:
    with fits.open(item) as hdus:
        ra = hdus[1].data['RA']
        dec = hdus[1].data['DEC']
    data['ra_min'].append(np.min(ra))
    data['ra_max'].append(np.max(ra))
    data['dec_min'].append(np.min(dec))
    data['dec_max'].append(np.max(dec))
    data['star_cat_file'].append(item)
df = pd.DataFrame(data)

df.to_parquet("star_catalog_index.parquet")
