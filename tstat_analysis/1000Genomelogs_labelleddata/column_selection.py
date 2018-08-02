#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import csv
from itertools import imap
from operator import itemgetter
import sys

delimiter = ','
inf=sys.argv[1]
outf='out-' + inf
with open(inf, 'rb') as input_file:
    reader = csv.reader(input_file, delimiter=delimiter)
    with open(outf, 'wb') as output_file:
        writer = csv.writer(output_file, delimiter=delimiter)
        writer.writerows(imap(itemgetter(45, 52, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105), reader))