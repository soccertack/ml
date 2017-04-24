import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
bnlearn = rpackages.importr('bnlearn')

# bif: bn.fit
bif = bnlearn.read_bif('alarm.bif')

print (bif)


