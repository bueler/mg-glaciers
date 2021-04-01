# temporary test program as I build the p-Laplacian smoother and problem

from meshlevel import MeshLevel1D
from smoothers.plap import PNGSPLap

# create a testargs object that mimics args returned by the ArgParse parser
class Namespace:
    '''Dummy class.'''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

testargs = Namespace(jacobi=False, omega=1.0, printwarnings=False, randomseed=1,
                     siaeta0=0.0, poissoncase='icelike', siashowsingular=False)

prob = PNGSPLap(testargs)
ml = MeshLevel1D(j=10, xmax=1.0)
prob.exactfigure(ml)
