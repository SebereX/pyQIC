"""
Test loading configurations from database and plotting them.
"""
import numpy as np
from qic import Qic

###################
# LOAD A CONFIGURATION
###################
stel = Qic.from_db('N3_082_575', db_path='/home/IPP-HGW/rodre/Documents/qi_database/')

#####################
# PLOT 
#####################
stel.plot_3d(r = 0.1, shaded = True, show = True)