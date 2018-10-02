import matplotlib.pyplot as plt
from matplotlib import rc


from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 

# on MAC OS

''' 
set defaults for nicely-formatted plots
define handy functions
'''
# plotting stuff
params = {'axes.labelsize': 26,
          'text.fontsize': 26,
          'legend.fontsize': 26,
          'xtick.labelsize': 26,
          'ytick.labelsize': 26,
          'text.usetex':True}

rc('text',fontsize=26)
rc('legend',fontsize=26)
rc('xtick',labelsize=26)
rc('ytick',labelsize=26)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


default_params = dict(nbins = 10,
                      steps = None,
                      trim = True,
                      integer = False,
                      symmetric = False,
                      prune = None)


