# coding: utf-8

# place this in the dir
# where train, dev and dev_2015 are there

import numpy as np
import sys
import collections
import argparse

# file to analyse
parser = argparse.ArgumentParser(
    description="Analyse the tags")
parser.add_argument("--file",
                    help="file to be analysed",
                    required=True)
parser.add_argument("--tag",
                    help="tag that is to be analysed", required=True)
parser.add_argument("--ref",
                    help="is the tag predicted or true",
                    choices=['true', 'predict'],
                    required=True)
args = parser.parse_args()

f = open(args.file, "r")
contents = f.read()
contents = contents.split("\n")

# entity type (GEO-LOC);
# or tag (B-GEO-LOC)
tag_type = args.tag

# reference
ref = args.ref

words = []
line_nos = []
predict_tags = []
true_tags = []

line = 1
for l in contents:
    if l != "":
        line_nos.append(line)
        words.append(l.split(" ")[0])
        true_tags.append(l.split(" ")[-2])
        predict_tags.append(l.split(" ")[-1])
    line += 1

# for each instance of the tag in the 
# true_tags print the entire line
print '\nListing all ' + ref + ' instances of ' + tag_type + ' tags in: ' + args.file
print ''

print ("{:>4} {:>20} {:>15} {:>15}").format("line", "word", "true", "predicted")
print ("{:>4} {:>20} {:>15} {:>15}").format("-"*4, "-"*20, "-"*15, "-"*15)
if ref == "predict":
    for i in range(len(predict_tags)):
        if predict_tags[i].lower() == tag_type.lower():
            print ("{:>4} {:>20} {:>15} {:>15}").format(
                line_nos[i],
                words[i],
                true_tags[i],
                predict_tags[i])
else:
    for i in range(len(predict_tags)):
        if true_tags[i].lower() == tag_type.lower():
            print ("{:>4} {:>20} {:>15} {:>15}").format(
                line_nos[i],
                words[i],
                true_tags[i],
                predict_tags[i])
