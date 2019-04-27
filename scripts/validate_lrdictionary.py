from __future__ import division
from argparse import ArgumentParser
import os
import json
import pickle
import io

def parse_dict(inFile, outFile):
    count = 0

    fout = io.open(outFile, 'w', encoding='utf8')

    with open(inFile) as f:
        for line in f:
            count += 1
            words = line.strip().split('\t')

            if len(words) < 2:
                continue

            fout.write(words[0] + '\t' + words[1] + '\n')

            if count%1000 == 0:
                print("Processed {} lines".format(count))

    print("Total Processed lines: {}".format(count))
    fout.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("-in", "--input", action='store', dest="inFile", type=str, required=True, help="input bilingual dictionary")
    parser.add_argument("-out", "--output", action='store', dest="outFile", type=str, required=True, help="input bilingual dictionary")

    args = parser.parse_args()
    outFile = args.outFile
    inFile = args.inFile

    assert os.path.isfile(inFile), \
        "Please check path of your input file!"

    parse_dict(inFile, outFile)

if __name__ == '__main__':
    main()