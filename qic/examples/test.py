from qic import Qic
import numpy as np
import matplotlib.pyplot as plt

stel = Qic.from_paper("QI NFP1 r2", nphi = 1001)

stel.plot()