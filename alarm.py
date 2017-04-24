import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
bnlearn = rpackages.importr('bnlearn')

# bif: bn.fit
bif = bnlearn.read_bif('alarm.bif')

# bn_result: bn
bn_result = bnlearn.bn_net(bif)

print(bn_result)


