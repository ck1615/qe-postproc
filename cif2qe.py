#!/usr/bin/env python3
import sys
import getopt
from ase.io import read, write

def cif2qe():

    task = 'scf'
    try:
        fname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")

    argv = sys.argv[2:]

    kwargs = {'format': 'espresso-in', 'crystal_coordinates': True}
    opts, args = getopt.getopt(argv, "t:k:o")
    for opt, arg in opts:
        if opt in ['-t', '--task']:
            if arg in ['scf', 'nscf', 'vc-relax', 'relax', 'md']:
                task = arg
            else:
                ValueError("Task {} is not recognised.".format(arg))
        elif opt in ['-k', '--k-points']:
            kwargs['kpoints'] = tuple(arg)
        elif opt in ['-o', '--kpoint-offset']:
            kwargs['koffset'] = tuple(arg)

    if '.cif' in fname:
        inputname = fname.replace(".cif", '.{}.in'.format(task))
    else:
        ValueError("File is not a .cif file")

    write(inputname, read(fname), **kwargs)

if __name__ == "__main__":
    cif2qe()
