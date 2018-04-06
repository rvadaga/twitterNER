# coding: utf-8

# place this in the dir
# where train, dev and dev_2015 are there

import numpy as np
import sys
import collections

f = open(sys.argv[1], "r")
con1 = f.read()

con1 = con1.split("\n")

tags = []

for l in con1:
    if l != "":
        tags.append(l.split("\t")[-1])

# find the frequency of
# each tag in the counter
counter = collections.Counter(tags)
counter_sorted = sorted(counter.items(), key=lambda x: x[0])

print '\n------------------------------'
print 'Listing tags in: ' + sys.argv[1]
for i in xrange(len(counter_sorted)):
    print i+1, counter_sorted[i][0], counter_sorted[i][1]

print '------------------------------\n'
