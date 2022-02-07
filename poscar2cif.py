#!/usr/bin/env python3

from ase.io import read, write
import sys
from findsymfile import findsym_wrap

def main():

    try:
        fname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")

    froot = fname.split('/')[:-1]
    cifname = '{}.cif'.format('.'.join(froot))
    print(cifname)
    write(cifname, read(fname))

    if sys.argv[2:] and sys.argv[2] == "-s":
        findsym_wrap(cifname, print_cif=True, axeso='abc', axesm='ab(c)',
                    lattol=1e-3, postol=1e-2)


if __name__ == "__main__":
    main()

