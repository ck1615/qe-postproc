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

class XML_Data:

    def __init__(self, xmlname):

        #Initialise file
        self.xmlname = xmlname
        self.xmltree = ET.parse(self.xmlname)

        #Important keywords as attributes (ntyp, nat, ibrav)
        self.ntyp = None
        self.nat = None
        self.ibrav = None
        self.alat = None
        self.nbnd = None

        #Structural parameters (unit cell, atomic positions)
        self.cell = []
        self.positions = {}
        self.scaled_positions = {}

        #Keywords
        self.keywords = {}
        self.xc = None

        #K-points, KS energy eigenvalues
        self.k_points = {}

        #Get data
        self.get_xml_data()


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
        self.recip_cell_xml = []
        for r in self.xmltree.findall\
                ("./output/basis_set/reciprocal_lattice/*"):
            self.recip_cell_xml.append([float(component) for component
                in r.text.split()])
        self.recip_cell_xml = np.array(self.recip_cell_xml)


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
        self.ntyp = int(self.xmltree.find('./output/atomic_species').\
                attrib['ntyp'])
        #nat & ibrav
        self.nat = int(self.xmltree.find('./output/atomic_structure').\
                attrib['nat'])
        self.ibrav = int(self.xmltree.find('./output/atomic_structure').\
                attrib['bravais_index'])
        self.alat = float(self.xmltree.find('./output/atomic_structure').\
                attrib['alat']) * Bohr


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
        self.nbnd = int(self.bands_keywords['nbnd'])


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

        self.k_points = {'crystal': [list of k-points in reciprocal cell
        coordinates], 'cartesian': [list of k-points in units of 1/Å]}

        In the .xml file, the k-points are giving in units of 2π/alat. This is
        converted by multiplication by 2π / alat.

        Let k_cryst and k_cart be the crystal coordinates and cartesian (1/a)
        k-point components. Let L and L* be the unit cell matrix (rows are
        lattice vectors) and the reciprocal cell matrix.

        L* = 2π*transpose(inv(L))

        We have: k_cryst * (L*) = k_cart [rows]
        Hence by inversion: k_cryst = k_cart * inv(L*)
                                    = k_cart * inv(2π*transpose(inv(L)))
                                    = k_cart * (1/2π) * transpose(L)


        Eigenvalues and occupations are shaped as (# k-points, 2*#bands)
        for spin-polarised case, but as (# k-points, #bands) in spin-un-
        -polarised case. 

        Hence I reshape it in the spin-polarised case as (2,len(kpts), nbnd)
        """

        #Get cartesian k-points used to sample the Brillouin Zone (either scf,
        #vc-relax or bands calculation) and convert into 1/Å from 2π/alat
        self.k_points['cartesian'] = np.array([[float(kval) for kval in \
                r.text.split()] for r in self.xmltree.findall(\
                './output/band_structure/ks_energies/k_point')]) * \
                2*np.pi / self.alat
        #Use unit cell to get crystal coordinates k-points 
        self.k_points['crystal'] = np.matmul(self.k_points['cartesian'],\
                (1/(2*np.pi)) * np.transpose(self.cell))

        self.eigvals = np.array([[float(eig) for eig in r.text.split()] for r \
                in self.xmltree.findall\
                ('./output/band_structure/ks_energies/eigenvalues')]) * \
                Hartree

        self.occupations = np.array([[float(occ) for occ in r.text.split()] \
                for r in self.xmltree.findall\
                ('./output/band_structure/ks_energies/occupations')])
        assert len(self.k_points['cartesian']) == len(self.eigvals) == \
                len(self.occupations), "The number of k-points doesn't match the "+\
                "number of lists of KS eigenvalues or occupancies."

        #Shape of eigenvalues
        eval_shape = self.eigvals.shape
        target_shape = (2, eval_shape[0], int(eval_shape[1]/2))

        #Treat spin polarised case
        if self.bs_keywords['lsda'] == 'true':
            self.eigvals = self.eigvals.reshape(target_shape)
            self.occupations = self.occupations.reshape(target_shape)


    def get_fermi_energy(self):
        """
        This function extracts the Fermi energy / energies
        """
        #Get Fermi energy or Fermi energies
        #Spin-polarised
        if self.bs_keywords['lsda'] == 'true':
            fermi_kw = 'two_fermi_energies'
        else:
            fermi_kw = 'fermi_energy'
        self.fermi_energy = [float(ef) * Hartree for ef in self.\
                    bs_keywords[fermi_kw].split()]



    def get_xml_data(self):
        """
        Get all data
        """

        #Structural data
        self.get_positions()
        self.get_cell()
        self.get_scaled_positions()

        #Keywords and calculation parameters
        self.get_control_variables()
        self.get_system_variables()
        self.get_xc_functional()
        self.get_bands_keywords()
        self.get_magnetisation_data()
        self.get_band_structure_keywords()

        #Get KS eigenvalues
        self.get_kpoint_eigenvalues()
        self.get_fermi_energy()





















