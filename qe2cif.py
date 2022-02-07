#!/usr/bin/env python3

from ase.io import read, write
import sys
from findsymfile import findsym_wrap

def main():

    try:
        fname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")

    if '.vc-relax.out' in fname:
        cifname = fname.replace('.vc-relax.out', '.cif')
    elif '.scf.out' in fname:
        cifname = fname.replace('.scf.out', '.cif')
    elif '.vc-relax.in' in fname:
        cifname = fname.replace('.vc-relax.in', '.cif')
    elif '.scf.in' in fname:
        cifname = fname.replace('.scf.in', '.cif')
    elif '.relax.out' in fname:
        cifname = fname.replace('.relax.out', '.cif')
    elif '.relax.in' in fname:
        cifname = fname.replace('.relax.in', '.cif')
    else:
        NameError("The output file provided contains neither 'scf.out' nor"+\
                " 'vc-relax.out', suggesting it is not an appropriate PWscf"+\
                " output file.")

    write(cifname, read(fname))
    if sys.argv[2:] and sys.argv[2] == "-s":
        findsym_wrap(cifname, print_cif=True, axeso='abc', axesm='ab(c)',
                     lattol=1e-2, postol=1e-2)


if __name__ == "__main__":
    main()

