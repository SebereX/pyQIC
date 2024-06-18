import numpy as np
import logging
from qic.qic import Qic
from qsc import Qsc

stel = Qic.from_paper(name = "5.2")
stel_QS = Qsc.from_paper(name = "5.2")
print(stel.r_singularity)
print(stel_QS.r_singularity)
stel.plot(show = False)
stel_QS.plot()