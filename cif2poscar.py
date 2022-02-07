#!/usr/bin/env python3
import sys
import getopt
from ase.io import read, write

def cif2poscar():

    try:
        fname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")

    argv = sys.argv[2:]

    kwargs = {'format': 'vasp', 'direct': True}
    opts, args = getopt.getopt(argv, "l:")
    for opt, arg in opts:
        if opt in ['-l', '--label']:
            kwargs['label'] = arg

    if '.cif' in fname:
        inputname = "POSCAR"
    else:
        ValueError("File is not a .cif file")

    write(inputname, read(fname), **kwargs)

if __name__ == "__main__":
    cif2poscar()
