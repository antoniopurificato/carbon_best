from .sequential.Caser import Caser, Caser2
from .sequential.CosRec import CosRec, CosRec2
from .sequential.HGN import HGN
from .sequential.GRU4Rec import GRU4Rec
# from NARM import NARM
# from NextItNet import NextItNet
# from NRMS import NRMS
from .sequential.SASRec import SASRec, SASRec2
from .sequential.BERT4Rec import BERT4Rec
from .sequential.NARM import NARM
from .sequential.CORE import CORE
# from S3Rec import S3Rec
#TODO: trovare metodo più intelligente imports

from ._version import __version__  # Import the '__version__' variable from this package

# The '__all__' variable can be used to specify which symbols are exported when
# someone uses 'from easy_exp import *'. However, the following symbols are commented out.
# You can uncomment them if you want them to be exported.

# __all__ = [
#     'pipeline',
#     "util_data",
#     "util_general",
#     '__version__'
# ]
