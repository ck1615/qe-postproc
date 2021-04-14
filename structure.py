#!/usr/bin/env python3
"""
This file contains classes which contain the input and output structures from a
quantum-espresso *name*.xml file.
"""

import xml.etree.ElementTree as ET
from glob import glob
from misctools import strindex
from ase.units import Bohr, Hartree
import numpy as np
from numpy.linalg import inv

class AtomicStructure:

    def __init__(self, xmlname):

        #Initialise file
        self.xmlname = xmlname
        self.xmltree = ET.parse(self.xmlname)

        #Important keywords as attributes (ntyp, nat, ibrav)
        self.ntyp = None
        self.nat = None
        self.ibrav = None

        #Structural parameters (unit cell, atomic positions)
        self.cell = []
        self.positions = {}
        self.scaled_positions = {}

        #Keywords
        self.keywords = {}
        self.xc = None


    def get_positions(self):
        """
        Get indexed element symbols and their absolute positions in Cartesian
        coordinates. 
        self.positions[(element, index)] = [x,y,z]
        """
        for r in self.xmltree.findall\
            ("./output/atomic_structure/atomic_positions/atom"):
            self.positions[(r.attrib['name'], int(r.attrib['index']))] = \
            [float(pos)*Bohr for pos in r.text.split()]


    def get_cell(self):
        """
        Get unit cell as a matrix where each ROW is a lattice vector

        """
        if self.cell != []:
           return None

        for r in self.xmltree.findall("./output/atomic_structure/cell/*"):
            self.cell.append([float(component)*Bohr for component in \
                    r.text.split()])
        #Convert to a numpy array
        self.cell = np.array(self.cell)
        self.reciprocal_cell = (2*np.pi) * np.transpose(inv(self.cell))


    def get_scaled_positions(self):
        """
        Get atomic positions in crystal coordinates.
        """

        #Ensure lattice vectors have been extracted
        if self.cell == []:
            self.get_cell()
        elif self.positions == {}:
            self.get_positions()

        for key in self.positions:
            self.scaled_positions[key] = np.matmul(self.positions[key], inv(\
                np.transpose(self.cell)))


    def get_control_variables(self):
        """
        Get control variables for this calculations
        """

        self.control_variables = {r.tag: r.text for r in self.xmltree.findall(\
                './input/control_variables/*')}


    def get_system_variables(self):
        """
        Get various structure variables:
            ntyp: number of types of elements
            nat : number of atoms
            nbnd: number of bands
            ibrav: Bravais lattice index

        """
        #ntyp
        self.ntyp = self.xmltree.find('./output/atomic_species').\
                attrib['ntyp']
        #nat & ibrav
        self.nat = self.xmltree.find('./output/atomic_structure').\
                attrib['nat']
        self.ibrav = self.xmltree.find('./output/atomic_structure').\
                attrib['bravais_index']


    def get_xc_functional(self):
        """
        Get the XC functional used.
        """
        self.xc = self.xmltree.find('./input/dft/functional').text


    def get_bands_keywords(self):
        """
        Get data relating to bands tag:
            tot_charge: total charge
            occupations: fixed or smearing
        """

        self.bands_keywords = {r.tag: r.text for r in self.xmltree.findall(\
                './input/bands/*')}


    def get_magnetisation_data(self):
        """
        Get data relating to the magnetization tag:
        lsda, noncolin, spinorbit, total, absolute
        """

        self.magnetisation_keywords = {r.tag: r.text for r in self.xmltree.findall(\
                './output/magnetization/*')}


    def get_energies(self):
        """
        Get total energy and all its contributions (etot, eband, ehart,
        vtxc, etxc, ewald)
        """

        self.total_energies = {r.tag: float(r.text)*Hartree for r in self.\
                xmltree.findall('./output/total_energy/*')}

    def get_band_structure_keywords(self):
        """
        Get data relating to the band_structure tag up until k-points are given
        """
        self.bs_keywords = {}
        for r in self.xmltree.findall("./output/band_structure/*"):
            if r.tag == 'starting_k_points':
                break
            else:
                self.bs_keywords[r.tag] = r.text

    def compute_band_gap(self):
        """
        Compute the band gap from the data
        """
        #CASE 1: Fixed occupancies
        #Check bands_keyword present
        try:
            occupations = self.bands_keywords['occupations']
        except AttributeError:
            self.get_bands_data()
            occupations = self.bands_keywords['occupations']

        if occupations == 'fixed':
            if ('highestOccupiedLevel' in self.bs_keywords.keys()) and \
            ('lowestUnoccupiedLevel' in self.bs_keywords.keys()):
                self.band_gap = Hartree*(float(self.bs_keywords[\
                        'lowestUnoccupiedLevel']) - float(self.bs_keywords[\
                        'highestOccupiedLevel'] ))
            else:
                KeyError("lowestUnoccupiedLevel not present since not enough"+\
                        " bands were used. Cannot compute band gap.")

    def get_kpoint_eigenvalues(self):
        """
        Get all the k-point eigenvalues as a dictionary or list of two dic-
        -tionary if lsda = true (spin-polarised calculation).

            kp_eigvals[(k1, k2, k3)] = [e1, e2, ..., eN]
        """

        #Create list of tuple of k-points
        k_points = [tuple([float(kval) for kval in r.text.split()]) for r\
                in self.xmltree.findall('./output/band_structure/ks_energies/k_point')]
        eigvals = [[float(eig) for eig in r.text.split()] for r in self.xmltree.\
                findall('./output/band_structure/ks_energies/eigenvalues')]
        occupations = [[float(occ) for occ in r.text.split()] for r in self.\
                xmltree.findall('./output/band_structure/ks_energies/occupations')]
        assert len(k_points) == len(eigvals) == len(occupations), "The " + \
                "number of k-points doesn't match number of lists of KS " + \
                "eigenvalues or occupancies."

        #Treat spin polarised case
        self.kpt_eigvals = []
        if self.bs_keywords['lsda'] == 'true':
            self.kpt_eigvals = [{},{}]
            nbnd = int(self.bands_keywords['nbnd'])
            for i, kpt in enumerate(k_points):
                self.kpt_eigvals[0][kpt] = eigvals[i][:nbnd]
                self.kpt_eigvals[1][kpt] = eigvals[i][nbnd:]
                assert eigvals[i] == self.kpt_eigvals[0][kpt]+ self.kpt_eigvals[1][kpt]



















