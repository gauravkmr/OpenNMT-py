from __future__ import division
from argparse import ArgumentParser
import os
import json
import pickle
import io

def parse_dict(inFile, outFile):
    count = 0

    fout = io.open(outFile, 'w', encoding='utf8')

    src_tgt_dict = {}

    with open(inFile) as f:
        for line in f:
            count += 1
            #words = line.rstrip().split()
            words = line.strip().split('\t')

            # print(words)

            if words[0] not in src_tgt_dict:
                src_tgt_dict[str(words[0])] = []

            map_list = []
            for i in range(1, len(words)):
                map_list.append(words[i])
                # fout.write(words[0] + '\t' + words[i].replace(' ', '_') + '\n')

            src_tgt_dict[str(words[0])] = map_list
            if count%1000 == 0:
                print("Processed {} lines".format(count))

    print("Total Processed lines: {}".format(count))

    # print(src_tgt_dict)
    # json_string = json.dumps(src_tgt_dict, ensure_ascii=False).encode('utf8')
    json_string = json.dumps(src_tgt_dict, ensure_ascii=False)
    fout.write(json_string)

    # with open(outFile, 'wb') as handle:
    #     pickle.dump(src_tgt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fout.close()

def main():
    parser = ArgumentParser()
    parser.add_argument("-in", "--input", action='store', dest="inFile", type=str, required=True, help="bilingual single src to multiple tgt word mapping")
    parser.add_argument("-out", "--output", action='store', dest="outFile", type=str, required=True, help="bilingual dict pickle dump")

    args = parser.parse_args()
    outFile = args.outFile
    inFile = args.inFile

    assert os.path.isfile(inFile), \
        "Please check path of your input file!"

    parse_dict(inFile, outFile)

if __name__ == '__main__':
    main()