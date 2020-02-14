#!/usr/bin/env python
#
# Copyright 2019 Dailer AG.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
Created on Tue May 10 16:01:02 2016

@author: Martin Manns
"""

import os
import hashlib


def duplicated_file_detection():

    file_hashes = {}

    BVH_PATH = r"C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\turnLeftRightStance"

    for root, dirs, files in os.walk(BVH_PATH):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath) as infile:
                hashres = hashlib.sha256(infile.read()).hexdigest()
                if hashres not in file_hashes:
                    file_hashes[hashres] = [name]
                else:
                    file_hashes[hashres] += [name]

    counter = 0
    for key in file_hashes:
        if len(file_hashes[key]) > 1:
            print(file_hashes[key])
            delete_duplicated_files(file_hashes[key], BVH_PATH)
            counter += 1

    print(counter, len(file_hashes))

def delete_duplicated_files(filenames, folder_path):
    """
    Delete the duplicated files in the list of filenames
    :param filenames: list of str
    :return:
    """
    segments = []
    for filename in filenames:
        segments.append(filename[:-4].split('_'))
    segments.sort(key=lambda x: int(x[4]))
    for seg in segments[:-1]:
        filename = '_'.join(seg) + '.bvh'
        if os.path.exists(os.path.join(folder_path, filename)):
            os.remove(os.path.join(folder_path, filename))

if __name__ == "__main__":
    duplicated_file_detection()