# takes an input file as cmd line argument
# contructs an orthographic representation
# for each word in the input file

import string
import sys

upperCase = string.ascii_uppercase
upperCaseOutput = "C" * len(upperCase)
lowerCase = string.ascii_lowercase
lowerCaseOutput = "c" * len(lowerCase)
punctuation = string.punctuation
punctuationOutput = "p" * len(punctuation)
digits = string.digits
digitsOutput = "n" * len(digits)

transTable = string.maketrans(upperCase + lowerCase + punctuation + digits,
                              upperCaseOutput + lowerCaseOutput +
                              punctuationOutput + digitsOutput)

inputFile = sys.argv[1]
print "Reading file " + inputFile

if ".txt" in inputFile:
    outputFile = inputFile.replace(".txt", "_orth.txt")
else:
    outputFile = inputFile + "_orth"

f_outputFile = open(outputFile, "w")
print "Writing to file " + outputFile
with open(inputFile, "r") as f:
    for line in f:
        line.decode('utf-8')
        if line.strip() == "":
            f_outputFile.write("\n")
        else:
            words = line.strip().split()
            f_outputFile.write(words[0].translate(transTable) + "\n")

f.close()
f_outputFile.close()
