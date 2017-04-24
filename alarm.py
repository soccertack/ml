import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
import pandas as pd
import sys

file_name='hw6_q3_data_50k.csv'

missing_columns = ["HISTORY", "CVP", "HYPOVOLEMIA", "LVFAILURE", "ERRLOWOUTPUT",
		"HRBP", "HREKG", "HRSAT", "INSUFFANESTH", "TPR", "KINKEDTUBE",
		"MINVOL", "FIO2", "SAO2", "PULMEMBOLUS", "SHUNT", "VENTMACH",
		"VENTTUBE", "VENTLUNG", "ARTCO2", "CATECHOL", "BP"]

pandas2ri.activate()

bnlearn = rpackages.importr('bnlearn')

# bif: bn.fit
bif = bnlearn.read_bif('alarm.bif')

# bn_result: bn
alarm_net = bnlearn.bn_net(bif)

###################################################################
# Get input data

if len(sys.argv) > 1:
	file_name = sys.argv[1]
else:
	print ("Use default 50k data")

with open(file_name) as myfile:
    count = sum(1 for line in myfile)
print ("Data size")
print (count -1)
###################################################################

###################################################################
# Generate one line of data

#Don't count the header
DATA_SIZE = count - 1

# sim: data.frame
sim = bnlearn.rbn(bif, DATA_SIZE)

sim_pan = pandas2ri.ri2py(sim)

sim_pan_extract = sim_pan[missing_columns]
#print (type(sim_pan))
#print(sim_pan_extract)
###################################################################

###################################################################
# Read the csv and concat with the generated data
data1 = ro.r['read.csv'](file_name)
data1_pan = pandas2ri.ri2py(data1)
#print (type(data1_pan))
#print (data1_pan)


#concat
result_oneline = pd.concat([data1_pan, sim_pan_extract], axis=1)
#print (type(result_oneline))
#print (result_oneline)
###################################################################

###################################################################
#Cross validation 

cv_result = bnlearn.bn_cv(result_oneline, alarm_net)
print (cv_result)
###################################################################

