import sys
sys.path.append("..")
from Index.produce_index import produce_index
f = produce_index(2**4, ssmpath = '/public/home/wufq/congyanping/Software/SSM/inSSM.hdf5')
f.plot_mean_index()
f.diffuse_x(freq = 1.)
