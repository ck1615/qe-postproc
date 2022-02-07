#!/usr/bin/env python3

from ase.io import read, write
import re
from glob import glob
import os


def main():
    """
    Getting energy
    """
    files = {re.search('La2CuO4_(.*).9.4.vc-relax.out', f).group(1): f for f in
            glob("*.vc-relax.out")}

    print(files)

    energies = {}

    for p in files:
        Atoms = read(files[p])
        energy = Atoms.get_potential_energy() /\
        Atoms.get_global_number_of_atoms()

        energies[p] = energy

    for p in energies:
        energies[p] -= energies['HTT']

    formula_unit = 7
    delta_E = formula_unit * 1e3 * (energies['LTO']-energies['LTT'])

    print(r"Energy difference = {:.3f} meV/(formula unit)".format(delta_E))



if __name__ == "__main__":
    main()



