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
from numpy.linalg import inv, norm
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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

        #Unit conversion
        if self.units == 'Hartree':
            conv = 1
        elif self.units == 'ase':
            conv = Hartree
        else:
            conv = 1

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
                ('./output/band_structure/ks_energies/eigenvalues')]) * conv

        self.occupations = np.array([[float(occ) for occ in r.text.split()] \
                for r in self.xmltree.findall\
                ('./output/band_structure/ks_energies/occupations')])
        assert len(self.k_points['cartesian']) == len(self.eigvals) == \
                len(self.occupations), "The number of k-points doesn't match the "+\
                "number of lists of KS eigenvalues or occupancies."

        #Treat spin polarised case
        if self.bs_keywords['lsda'] == 'true':
            #Shape of eigenvalues
            eval_shape = self.eigvals.shape
            target_shape = (2, eval_shape[0], int(eval_shape[1]/2))
            #Reshape eigenvalues
            self.eigvals = self.eigvals.reshape(target_shape)
            self.occupations = self.occupations.reshape(target_shape)


    def get_fermi_energy(self):
        """
        This function extracts the Fermi energy / energies
        """
        #Get Fermi energy keyword depending on spin polarisation
        if self.bs_keywords['lsda'] == 'true':
            fermi_kw = 'two_fermi_energies'
        else:
            fermi_kw = 'fermi_energy'
        #Get Fermi energy(ies)
        self.fermi_energy = [float(ef) * Hartree for ef in self.\
                    bs_keywords[fermi_kw].split()]

        self.eigvals[0] -= self.fermi_energy[0]
        self.eigvals[1] -= self.fermi_energy[1]

    def band_structure(self, figsize=(6,6), energy_window=40):
        return self.BandStructure(self, figsize=figsize, energy_window=\
                energy_window)


    class BandStructure:
        """
        This class contains all the methods and attributes for a band structure
        plot
        """

        def __init__(self, xml_data, figsize=(6,6), energy_window=10):

            #self.tolerance:
            self.tol = 1e-10
            #Instantiate outer class XML_Data in the inner class 
            self.xml_data = xml_data
            self.bands_inputfile = self.xml_data.xmlname.replace('xml',\
                    'bands.in')

            #Plot characteristics
            self.figsize = figsize
            self.spin_up_colour = 'b--'
            self.spin_down_colour = 'r'
            self.xlim = (0,1)
            self.xlabel = 'Wavevectors'
            self.ylabel = r'$E - E_{\mathrm{Fermi}}$ / eV'
            self.y_majorticks = 1
            self.y_minorticks = 0.5
            self.y_major_tick_formatter = '{x:.0f}'
            self.energy_window = energy_window

            #Bands and path related variables
            self.kpath_idx = []   #Indices from 0.0 to 1.0 of k-points
            self.path = None
            self.path_ticks = None
            self.labels = []


        def get_highsym_data(self):
            """
            Gets all data relative to the high-symmetry points in the Brillouin
            zone required to perform the plot.
            """
            self.get_kpath_indices()
            self.get_highsym_kpoints()
            self.get_highsym_ticks()
            self.get_highsym_points_labels()
            return None


        def get_kpath_indices(self):
            """
            This function takes a set of k-points and associates to it a list of
            non-negative real numbers corresponding to the distance from the
            initial k-point on the path.
            """
            path = []
            for i, kpt in enumerate(self.xml_data.k_points['cartesian']):
                if i == 0:
                    path.append(0.0)
                else:
                    path.append(path[i-1] + norm(kpt - \
                            self.xml_data.k_points['cartesian'][i-1]))

            #Normalise list between 0.0 and 1.0
            self.kpath_idx = [(idx - path[0]) / (path[-1] - path[0]) for idx \
                    in path]


        def get_highsym_kpoints(self):
            """
            Gets the high-symmetry points used to perform the band structure
            calculation
            """
            #Open bands input file
            with open(self.bands_inputfile, 'r+') as f:
                self.input_lines = f.readlines()

            #Get start and end indices
            start_idx = strindex(self.input_lines, "K_POINTS crystal")
            end_idx = strindex(self.input_lines, "ATOMIC_POSITIONS")

            #Extract path in crystal coordinates
            self.path = np.array([[float(l) for l in line.split()[:-1]] for \
                    line in self.input_lines[start_idx + 2:end_idx] if \
                    line.split() != [] ])


        def get_highsym_ticks(self):
            """
            This function gets the locations of the high-symmetry points along
            the x-axis of the plot.
            """

            #Define high-symmetry point ticks: 
            self.path_ticks = np.zeros(len(self.path))

            #Ensure first and last high-symmetry point correspond with start
            #and end of k-point list
            assert norm(self.path[0] - self.xml_data.k_points['crystal'][0]) <\
            self.tol and norm(self.path[-1] - self.xml_data.k_points['crystal']\
                    [-1]) < self.tol, "Initial and final are not what is expected"

            #Set the values of the first and last ticks
            self.path_ticks[0] = 0.0
            self.path_ticks[-1] = 1.0

            #Initial k-point index
            kpt_idx = 1
            #Iterate over non-extremal high-symmetry points
            for ip, p in enumerate(self.path[1:-1]):
                #Iterate over k-points
                for ik, k in enumerate(self.xml_data.k_points['crystal']):
                    #Only consider k-points above index
                    if ik < kpt_idx:
                        continue
                    if norm(k-p) < self.tol:
                        kpt_idx = ik + 1 #Start at next k-point after match
                        self.path_ticks[ip + 1] = self.kpath_idx[ik]
                        break


        def get_highsym_points_labels(self):
            """
            This function returns the labels used when plotting the band struc-
            -ture.
            """
            for ip, p in enumerate(self.path):
                #Gamma point
                if norm(p) < self.tol:
                    self.labels.append(r"\Gamma")
                else:
                    self.labels.append(tuple(p))


        def plot_band_structure(self):
            """
            This function plots the band structure
            """
            fig, ax = plt.subplots(figsize=self.figsize)
            #Spin polarised case
            if self.xml_data.bs_keywords['lsda'] == 'true':
                ax.plot(self.kpath_idx, self.xml_data.eigvals[0], self.\
                        spin_up_colour, label='Spin up', linewidth=1.0)
                ax.plot(self.kpath_idx, np.flip(self.xml_data.eigvals[1],\
                        axis=0), self.spin_down_colour, label='Spin down',\
                        linewidth=1.0)
            else:
                ax.plot(self.kpath_idx, self.xml_data.eigvals, self.\
                        spin_up_colour)

            #Set energy (y) axis quantities
            ylim = (-self.energy_window / 2, self.energy_window / 2)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MultipleLocator( self.y_majorticks ))
            ax.yaxis.set_major_formatter(self.y_major_tick_formatter)
            ax.yaxis.set_minor_locator(MultipleLocator( self.y_minorticks ))
            ax.set_ylabel( self.ylabel )

            #Set high-symmetry point quantities
            ax.set_xlim((0.0,1.0))
            ax.set_xticks(self.path_ticks)
            ax.set_xticklabels(self.labels, rotation=45, fontsize=10)

            #Plot vertical lines at each high-symmetry point
            for i, t in enumerate(self.path_ticks):
                ax.axvline(x=t, c='k', linewidth=1)

            #Save figure
            fig.tight_layout()
            #fig.savefig(self.xml_data.replace('xml', 'pdf'))

            return None






















