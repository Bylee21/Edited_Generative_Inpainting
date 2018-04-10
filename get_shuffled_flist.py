#!/usr/bin/python

import argparse
import os
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./training_data/GOPR47/training', type=str,
                    help='The folder path')
parser.add_argument('--output_filename', default='./data_flist/train_shuffled.flist', type=str,
                    help='The output filename.')
parser.add_argument('--is_shuffled', default='1', type=int,
                    help='Needed to shuffle')

if __name__ == "__main__":

    args = parser.parse_args()

    # Open a file
    dirs = os.listdir(args.folder_path)

    file_names = []
    # This would print all the files and directories
    for file in dirs:
        file = args.folder_path + "/" + file
        file_names.append(file)

    # shuffle file names if set
    if args.is_shuffled == 1:
        shuffle(file_names)

    # make output file if not existed
    if not os.path.exists(args.output_filename):
        os.mknod(args.output_filename)

    # write to file
    fo = open(args.output_filename, "w")
    fo.write("\n".join(file_names))
    fo.close()

    # print process
    for e in file_names:
        print(e)
    print("Written file is: ", args.output_filename, ", is_shuffle: ", args.is_shuffled)


