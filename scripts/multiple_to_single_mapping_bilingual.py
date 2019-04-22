from __future__ import division
from argparse import ArgumentParser
import os

def parse_dict(inFile, outFile):
    count = 0

    fout = open(outFile, 'w')

    with open(inFile) as f:
        for line in f:
            count += 1
            #words = line.rstrip().split()
            words = line.strip().split('\t')

            for i in range(1, len(words)):
                fout.write(words[0] + '\t' + words[i].replace(' ', '_') + '\n')

            if count%1000 == 0:
                print("Processed {} lines".format(count))

    print("Total Processed lines: {}".format(count))
    fout.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("-in", "--input", action='store', dest="inFile", type=str, required=True, help="Multiple mapping bilingual dictionary")
    parser.add_argument("-out", "--output", action='store', dest="outFile", type=str, required=True, help="Single mapping bilingual dictionary")

    args = parser.parse_args()
    outFile = args.outFile
    inFile = args.inFile

    assert os.path.isfile(inFile), \
        "Please check path of your input file!"

    parse_dict(inFile, outFile)

if __name__ == '__main__':
    main()