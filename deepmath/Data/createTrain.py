from __future__ import print_function, division
import csv
import numpy as np

def compose(f):
    '''
    Composes one function inside another
    '''
    if type(f) is list:
        f = f[0]

    for i in xrange(f.count('@')):
        insert = np.random.randint(0,num_of_funcs-1)
        print(functions[insert])
        f = f.replace('@', functions[insert])
    return f

def addVariables(func, var):
    return func.replace('@', var)

# ------------------------------------------------------------------------------- #
# Constants
n=50 # num of data points
inFile = 'functions.txt'
outFile = "train.csv"

#Read functions into list
#TODO Do we want to weight the probabilities of these functions to make them better
functions = []
with open(inFile, 'rb') as funccsv:
    functionsreader = csv.reader(funccsv, delimiter=',')
    for row in functionsreader:
        if row:
            functions.append(row[0])

num_of_funcs = len(functions)

outWriter = open(outFile, 'w')
#Get Random number for number of compositions
for i in xrange(n):
    print('\n----- Step {} ----\n'.format(i))
    comp_num = np.random.randint(0,4)
    insert = np.random.randint(0,num_of_funcs-1)
    func = functions[insert]
    for j in xrange(comp_num):
        print("{} Before: ".format(j), func)
        func = compose(func)
        print("{} After: ".format(j), func)
    func = addVariables(func, 'x')
    print("FINAL: ", func)
    outWriter.write("{}, {}\n".format(i+1, func))
outWriter.close()


    #TODO Do we want to weight the probabilities of these  to make them more realistic

    #for each composition draw a random function and compose it with the others


#Decide if definite or indefinite

# if definite, draw bounds
