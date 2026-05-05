import requests
import shutil
import json
import numpy as np
from qic import Qic


config = 'N2_427_0169'

# Load config from online database
stel_online = Qic.from_db(config, online = True)
stel_online.plot_3d(r = 0.1, shaded = True, show = True)

# Load from local database
db_path='/home/../mnt/d/Research/Stellerator/Datsshare_sync/DB/Paper_scripts/'
stel_local = Qic.from_db(config, online = False, db_path=db_path)
stel_local.plot_3d(r = 0.1, shaded = True, show = True)

