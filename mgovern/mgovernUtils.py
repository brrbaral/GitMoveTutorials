# required functions
from math import e, exp, fabs, modf, log as ln
from pprint import pprint

#--- constants ----------------------------------------------------------------
# constants
infinity = 9e999
ln2 = ln(2)
binary = 1/ln2
natural = 1/ln(2.71828182846)
decimal = 1/ln(10)


 #--- functions ----------------------------------------------------------------
    # logarithm with configurable base
def log(self,x,*logtype):
  try:
    basefactor = logtype[0]
  except:
    basefactor = 1
  return ln(x) * basefactor
   # '''application example:
    # print log (16,(binary)) # yields 4.0
    # print log(16) # yields 2.77258872224
    # print log (16,(decimal)) # yields 1.20411998266

def error_if_not_in_range01(self,value):
  if (value < 0) or (value > 1): # check range
              #print
    raise Exception(str(value) + ' is not in [0,1)!')

    #print(error_if_not_in_range01(0.5))


def error_if_not_1(self,value):
  if (value != 1):
  # not 100%
    raise Exception('Sum ' + str(value) + ' is not 1!')

def sum_of_table(self,x):
  sum = 0.0
  for x_i in x:
    #print(x_i)
    error_if_not_in_range01(x_i)
    sum += x_i
    #print(x_i)
    #error_if_not_1(sum)
  return sum