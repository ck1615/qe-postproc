#!/usr/bin/env python3
"""
This file contains functions for generating an input file for the correct ibrav
as the relaxed output file for PWscf calculations
"""
import sys
import getopt
import os
from ase.io import read, write
from misctools import strindex, strindices as stri
from ase.units import Bohr
import math as m
import numpy.linalg as la
import numpy as np
from copy import deepcopy

def get_cmdline_options():
    """
    This function parses the command line options to get the input filename and
    any options requested
    """

    #Define PWscf calculation types: 
    pwscf_calctypes = ['scf', 'nscf', 'bands', 'relax', 'vc-relax', 'md','vc-md']

    try:
        seed = sys.argv[1]
    except IndexError:
        IndexError("No filename has been provided.")
    argv = sys.argv[2:]

    #Iterate through options
    kwargs = {}
    opts, args = getopt.getopt(argv, "c:rt:")
    for opt, arg in opts:
        if opt in ['-c', '--calculation']:
            kwargs['calc_type'] = arg
            assert arg in pwscf_calctypes,\
            "Chosen calculation type: {}, is not one of the options for PWscf."
        elif opt in ['-r', '--rotate-cell']:
            kwargs['rotate'] = True
        elif opt in ['-t', '--rotation-angle']:
            kwargs['theta'] = float(arg)
        else:
            Warning("Calculation type not specified, we'll assume to use the"+\
                    "same as what was in the output file.")

    return seed, kwargs

class QEOutput:


    def __init__(self, seed, calc_type='scf', rotate=False, theta=0.0):

        self.calc_type = calc_type
        self.outputfile = seed
        self.Atoms = read(self.outputfile)
        self.inputfile = self.outputfile.replace("out", "in")
        self.celldms = None
        self.elempos = None
        self.input_lines = None
        self.relaxed_lines = None
        self.final_fname = None
        self.rotate = rotate
        self.theta = theta
        self.tol = 1e-10
        return None


    def get_new_structure(self):

        #Get elements as named in the quantum-espresso input file
        if self.input_lines is None:
            self.get_input_lines()

        old_atompos_idx = strindex(self.input_lines, \
                "ATOMIC_POSITIONS crystal")

        #Assume ATOMIC_POSITIONS is LAST part of input file
        atompos_lines = self.input_lines[old_atompos_idx + 1:]
        elems = [line.split()[0] for line in atompos_lines]

        if self.rotate:
            #Get rotated unit cell matrix and the positions
            self.celldms, P2 = self.rotate_unit_cell()

            #Ensure number of positions is equal to number of elements
            assert len(elems) == len(P2), "Number of atoms ({}) is not equal "+\
                "to number of atomic positions, {}".format(len(elems), \
                len(P2))

            self.elempos = {tuple(P2[i]): elem for i, elem in \
                    enumerate(elems)}

        else:
            #Get lattice parameters only
            self.celldms = [l/Bohr if i in range(3) else l for i,l in \
                    enumerate(self.Atoms.cell.cellpar())]

            posns = self.Atoms.get_scaled_positions()
            self.elempos = {tuple(pos): elems[i] for i, pos in \
                    enumerate(posns)}

        return None


    def get_input_lines(self):

        with open(self.inputfile, "r+") as f:
            self.input_lines = f.readlines()

        return None


    def replace_structure(self):

        #Update lattice parameters (in Bohr)
        if self.rotate:
            lattice_updated_lines = self.replace_rotated_lattice_params()
        else:
            lattice_updated_lines = self.replace_lattice_params()

        #Update scaled positions
        relaxed_lines = self.replace_scaled_positions(lattice_updated_lines)

        return relaxed_lines


    def rotate_unit_cell(self):
        """
        This function rotates the unit cell and modifies the cell and positions
        accordingly. This function returns the unit cell matrix in Bohr and
        the atomic positions in crystal coordinates
        """
        #Convert to radians
        assert self.theta != 0, "Angle is 0.0°, no rotation required."
        t = self.theta * (180 / m.pi)

        #Define (passive) transformation matrix
        T = np.array([ [m.cos(t), m.sin(t), 0.0],
            [-m.sin(t), m.cos(t), 0.0],
            [0.0, 0.0, 1.0]
            ])
        assert abs(la.det(T) - 1) < self.tol, "Determinant of transformation "+\
                "matrix is not unity: {}".format(la.det(T))

        #Get unit cell and scaled positions
        L = np.array(self.Atoms.cell)
        P = np.array(self.Atoms.get_scaled_positions())

        #Get rotated unit cell
        L2 = np.matmul(L, np.transpose(T))
        P2 = np.matmul(P, np.matmul(L, np.matmul(T, la.inv(L))))

        assert abs(la.det(L2) - la.det(L)) < self.tol, "Determinant of unit "+\
                "cell matrix is not conserved. Original: {}, Rotated: {}".\
                format(la.det(L), la.det(L2))

        assert la.norm(np.matmul(P2, L2) - np.matmul(P, L)) < self.tol, \
                "Cartesian coordinates of atoms has changed via transformation."
        return L2, P2


    def get_ibrav(self):
        if self.input_lines is None:
            self.get_input_lines()

        ibrav_idx = strindex(self.input_lines, "ibrav", first=True)
        self.ibrav = int(self.input_lines[ibrav_idx].split()[2].strip(","))

        return None


    def replace_rotated_lattice_params(self):

        #Get positions of celldm()
        celldm_ids = stri(self.input_lines, "celldm")

        #Deep copy input lines to modify
        input_lines = deepcopy(self.input_lines)

        #Iterate through celldm indices, replace first occurrence and delete
        #all others
        for i, idx in enumerate(celldm_ids):
            if i == 0:
                input_lines[idx] = '  ibrav = 0\n'
            else:
                input_lines.pop(idx)

        #Insert before ATOMIC_POSITIONS the new cell parameters
        atom_pos_idx = strindex(input_lines, 'ATOMIC_POSITIONS')

        input_lines.insert(atom_pos_idx, 'CELL_PARAMETERS bohr\n')
        count=1
        for v in self.celldms:
            input_lines.insert(atom_pos_idx + count, \
                    '{:.8f} {:.8f} {:.8f}\n'.format(v[0]/Bohr, v[1]/Bohr, \
                    v[2]/Bohr))
            count += 1
        input_lines.insert(atom_pos_idx + count, '\n')

        return input_lines


    def replace_lattice_params(self):

        #Get ibrav
        self.get_ibrav()

        #Get cell dimension indices required for given self.ibrav
        if self.ibrav in [6, 7] :
            idx_list = [1,3]
            self.celldms[2] /= self.celldms[0]
        elif self.ibrav == 8:
            idx_list = [1,2,3]
            self.celldms[1] /= self.celldms[0]
            self.celldms[2] /= self.celldms[0]
        elif self.ibrav == 1:
            idx_list = [1]

        for i in idx_list:
            try:
                celldm_idx = strindex(self.input_lines, "celldm({})".format(i))
            except:
                raise ValueError("Need spaces at either side of equal sign"+\
                        " for cell dimensions.")

            old_celldm = self.input_lines[celldm_idx].partition("celldm({})".\
                    format(i))[-1].split()[1]
            if old_celldm[-1] == ",":
                old_celldm = old_celldm[:-1]

            self.input_lines[celldm_idx] = self.input_lines[celldm_idx].\
                    replace(old_celldm, "{}d0".format(str(self.celldms[i - 1])))

        return self.input_lines


    def replace_scaled_positions(self, lattice_updated_lines):
        #Get index for atomic position
        atom_pos_idx = strindex(lattice_updated_lines, "ATOMIC_POSITIONS crystal")

        #Assume ATOMIC_POSITIONS is LAST part of input file
        lattice_updated_lines = lattice_updated_lines[:atom_pos_idx + 1]

        for pos, elem in self.elempos.items():
            lattice_updated_lines.append('{} {:.15f} {:.15f} {:.15f}\n'.format(elem, pos[0],\
                    pos[1], pos[2]))

        return lattice_updated_lines


    def modify_calc_type(self, final_lines):

        idx_calc_type = strindex(final_lines, "calculation=")
        final_lines[idx_calc_type] = "  calculation='{}'\n".format(self.calc_type)

        #Change name to reflect calculation name
        init_calc_type = self.inputfile.split(".")[-2]

        if self.rotate:
            self.final_fname = self.inputfile.replace(init_calc_type,
                    "relaxed.rot{:.0f}.{}".format(self.theta,self.calc_type))
        else:
            self.final_fname = self.inputfile.replace(init_calc_type,
                "relaxed.{}".format(self.calc_type))

        return final_lines


    def print_relaxed_structure(self):

        #Define input file and get line
        self.get_input_lines()

        #Get celldms & elempos
        self.get_new_structure()

        #Replace structure in input_lines
        final_lines = self.replace_structure()

        #Change calculation type
        final_lines = self.modify_calc_type(final_lines)

        #Get the Mg-O bond length
        #mgo = get_mgo_bond_length(celldms, elempos)

        #Implement changes for distance-based constraints
        #final_lines = distance_constraints(relaxed_lines, mgo)

        #Print new file
        with open(self.final_fname, "w+") as f:
            for line in final_lines:
                f.write(line)


def main():

    seed, kwargs = get_cmdline_options()

    Structure = QEOutput(seed, **kwargs)
    Structure.print_relaxed_structure()

if __name__ == "__main__":
        main()


