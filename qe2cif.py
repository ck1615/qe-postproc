#!/usr/bin/env python3

from ase.io import read, write
import sys

def main():

    try:
        fname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")

    if '.vc-relax.out' in fname:
        cifname = fname.replace('.vc-relax.out', '.cif')
    elif '.scf.out' in fname:
        cifname = fname.replace('.scf.out', '.cif')
    else:
        NameError("The output file provided contains neither 'scf.out' nor"+\
                " 'vc-relax.out', suggesting it is not an appropriate PWscf"+\
                " output file.")

    write(cifname, read(fname))



if __name__ == "__main__":
    main()

