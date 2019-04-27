from __future__ import division
from argparse import ArgumentParser

import xml.etree.ElementTree as ET
import os

def parse_and_output(lang, prefix, filename, tags, outputDir):
    tree = ET.parse(filename)
    root = tree.getroot()

    for tag in tags:
        lines = root.findall('.//' + tag)
        with open(outputDir + lang + '_' + prefix + '_' + tag + '.txt', 'w') as f:
            for line in lines:
                f.write(line.text + '\n')

def get_tag_list(tags):
    tags_l = tags.strip().split(",")
    tags = []
    for t in tags_l:
        tags.append(t + '_SOURCE')
        #tags.append(t + '_TARGET')
    return tags

def main():
    parser = ArgumentParser()
    parser.add_argument("-lang", "--language", action='store', dest="language", type=str, required=True,
                        help="Source language being translated.")
    parser.add_argument("-prefix", "--prefix", action='store', dest="prefix", type=str, required=True,
                        help="Prefix should be 'train', 'dev', or 'test'")
    parser.add_argument("-in", "--input", action='store', dest="filename", type=str, required=True, help="XML file")
    parser.add_argument("-tags", "--tags", action='store', dest="tags", type=str, required=True, help="comma-separated tagnames")
    parser.add_argument("-out", "--output", action='store', dest="outputDir", type=str, required=True, help="output data directory")

    args = parser.parse_args()
    prefix = args.prefix.lower()
    if prefix not in ['train', 'dev', 'test', 'eval']:
        print("Prefix should be 'train', 'dev', 'test', or 'eval'")
        exit(1)

    lang = args.language
    filename = args.filename
    outputDir = args.outputDir
    tags_list = get_tag_list(args.tags)

    assert os.path.isfile(filename), \
        "Please check path of your input file!"

    parse_and_output(lang, prefix, filename, tags_list, outputDir)


if __name__ == '__main__':
    main()
