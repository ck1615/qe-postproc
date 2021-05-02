#!/usr/bin/env python3
"""
This file contains classes which contain the input and output structures from a
quantum-espresso *name*.xml file.
"""

#XML Parsing tools
import xml.etree.ElementTree as ET
from misctools import strindex

#Data structures and calculating auxiliary quantities
from ase.units import Bohr, Hartree
import numpy as np
from numpy.linalg import inv, norm

#Font
#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#    "font.sans-serif": ["Palatino"]})

class XML_Data:

    def __init__(self, xmlname, units='ase'):

        #Initialise file
        self.xmlname = xmlname
        self.xmltree = ET.parse(self.xmlname)

        #Units (ase: Å, eV, Pa, Hartree: Hartree, Bohr, ...)
        self.units = units

        #Important keywords as attributes (ntyp, nat, ibrav)
        self.ntyp = None
        self.nat = None
        self.ibrav = None
        self.alat = None
        self.nbnd = None

        #Structural parameters (unit cell, atomic positions)
        self.cell = np.array([])
        self.positions = {}
        self.scaled_positions = {}

        #Keywords
        self.keywords = {}
        self.xc = None

        #K-points, KS energy eigenvalues
        self.k_points = {}

        #Get data
        self.get_xml_data()


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

        #Get number of bands
        try:
            self.nbnd = int(self.bands_keywords['nbnd'])
        except KeyError:
            self.nbnd = int(self.bs_keywords['nbnd'])

        #Get KS eigenvalues
        self.get_kpoint_eigenvalues()


    def get_distance(self, i1, i2):
        """
        This function gets the absolute distance between two atoms indexed by
        the indices i1 and i2.
        """
        key_list = list(self.positions)
        key1, key2 = key_list[i1 - 1], key_list[i2 - 1]

        return norm(np.array(self.positions[key1]) -\
                np.array(self.positions[key2]))


    def get_angle(self,i1,i2,i3):
        """
        This function returns the angle between the atoms with indices i1, i2
        and i3 where i1 is the atom with respect to which the angle is compu-
        ted.

        The third index can also be one of x, y, z, xy, yz, xz and the angle
        will be computed between the vector i2 - i1 and the corresponding axis
        specified by i3. In this case, i3 is a string.

        """
        key1, key2 = list(self.positions)[i1 - 1], list(self.positions)[i2 - 1]
        vec1 = np.array(self.positions[key2]) - np.array(self.positions[key1])

        direction_dict = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]),
                'z': np.array([0,0,1]), 'xy': np.array([1,1,0]),
                'yz': np.array([0,1,1]), 'xz': np.array([1,0,1])}

        if i3 in direction_dict.keys():
            vec2 = direction_dict[i3]
        else:
            key3 = list(self.positions)[i3 - 1]
            vec2 = np.array(self.positions[key3]) - np.array(self.positions[key1])

        return (180 / np.pi) * np.arccos(vec1.dot(vec2) / \
                (norm(vec1) * norm(vec2)))


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
        if self.cell.size:
           return None
        cell = []
        for r in self.xmltree.findall("./output/atomic_structure/cell/*"):
            cell.append([float(component)*Bohr for component in \
                    r.text.split()])
        #Convert to a numpy array
        self.cell = np.array(cell)

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
        if self.cell.size:
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
        xc_dict = {'MGGA_X_R2SCAN MGGA_C_R2SCAN': 'R2SCAN'}
        self.xc = self.xmltree.find('./input/dft/functional').text

        if self.xc in xc_dict:
            self.xc = xc_dict[self.xc]


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


    def get_spin_splitting(self):
        """
        This function returns the average difference between the eigenvalues
        of spin up and spin down channels
        """

        assert self.nbnd == \
                int(self.bs_keywords['nbnd_up']) == \
                int(self.bs_keywords['nbnd_dw']),\
                "The spin-splitting can only be measured in cases where the"+\
                "number of spin-up and spin-down bands are the same."

        eigval_diff = abs(self.eigvals[:, :self.nbnd] -
                self.eigvals[:, self.nbnd:])
        self.mean_spin_splitting = np.sum(eigval_diff) / eigval_diff.size
        self.max_spin_splitting = np.max(eigval_diff)


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

        #Unit conversion
        if self.units == 'Hartree':
            conv = 1
        elif self.units == 'ase':
            conv = Hartree
        else:
            conv = 1

        #Get cartesian k-points used to sample the Brillouin Zone (either scf,
        #vc-relax or bands calculation) and convert into 1/Å from 2π/alat
        self.k_points['tpiba'] = np.array([[float(kval) for kval in \
                r.text.split()] for r in self.xmltree.findall(\
                './output/band_structure/ks_energies/k_point')])
        self.k_points['cartesian'] = self.k_points['tpiba'] * (2*np.pi / self.alat)
        #Use unit cell to get crystal coordinates k-points 
        self.k_points['crystal'] = np.matmul(self.k_points['cartesian'],\
                (1/(2*np.pi)) * np.transpose(self.cell))

        self.eigvals = np.array([[float(eig) for eig in r.text.split()] for r \
                in self.xmltree.findall\
                ('./output/band_structure/ks_energies/eigenvalues')]) * conv

        self.occupations = np.array([[float(occ) for occ in r.text.split()] \
                for r in self.xmltree.findall\
                ('./output/band_structure/ks_energies/occupations')])
        assert len(self.k_points['cartesian']) == len(self.eigvals) == \
                len(self.occupations), "The number of k-points doesn't match the "+\
                "number of lists of KS eigenvalues or occupancies."

        #Get Fermi energy and shift eigenvalues accordingly
        self.get_fermi_energy()
        self.eigvals -= self.fermi_energy[0]


    def get_fermi_energy(self):
        """
        This function extracts the Fermi energy / energies, but uses the
        highest occupied level when fixed occupancies are used
        """

        #Use highest occupied level for Fermi energy if occupations fixed
        #if True:
        if self.bands_keywords['occupations'] == 'fixed':
            fermi_kw = 'highestOccupiedLevel'
            self.fermi_energy = [float(ef) * Hartree for ef in self.\
                    bs_keywords[fermi_kw].split()]
        else:
            try:
                fermi_kw = 'fermi_energy'
                self.fermi_energy = [float(ef) * Hartree for ef in self.\
                       bs_keywords[fermi_kw].split()]
            except KeyError:
                fermi_kw = 'two_fermi_energies'
                self.fermi_energy = [float(ef) * Hartree for ef in self.\
                        bs_keywords[fermi_kw].split()]


def main():
    return None



if __name__ == "__main__":
    main()








