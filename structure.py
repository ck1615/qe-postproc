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

#Plotting
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#Parsing command line options
import sys
import getopt

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

        #Get KS eigenvalues
        self.get_kpoint_eigenvalues()


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


    def band_structure(self, figsize=10, energy_window=10):
        """
        Instantiates the BandStructure inner class with optional arguments.
        """
        if self.control_variables['calculation'] != 'bands':
            Warning("The xml file is written after a {} calculation, which "+\
                    "is not a bands calculation, required for the "+\
                    "production of a band structure.")

        return self.BandStructure(self, figsize=figsize, energy_window=\
                energy_window)


    class BandStructure:
        """
        This class contains all the methods and attributes for a band structure
        plot
        """

        def __init__(self, xml_data, figsize=10, energy_window=10):

            #self.tolerance:
            self.tol = 1e-10
            #Instantiate outer class XML_Data in the inner class 
            self.xml_data = xml_data
            self.bands_inputfile = self.xml_data.xmlname.replace('xml',\
                    'bands.in')

            #Plot characteristics
            self.figsize = figsize
            self.spin_down_colour = ':ro'
            self.spin_up_colour = '--bo'
            self.markersize = 2
            self.linewidth = self.markersize / 3
            self.xlim = (0,1)
            self.xlabel = 'Wavevectors'
            self.ylabel = r'$E - E_{\mathrm{Fermi}}$ / eV'
            self.y_majorticks = 1
            self.y_minorticks = 0.5
            self.y_major_tick_formatter = '{x:.0f}'
            self.energy_window = energy_window

            #Band gap characteristics
            self.HO = None
            self.LU = None
            self.band_gap = None
            self.nocc = int(np.ceil(float(self.xml_data.bs_keywords['nelec'])/2))

            #Bands and path related variables
            self.kpath_idx = []   #Indices from 0.0 to 1.0 of k-points
            self.path = None
            self.path_ticks = None
            self.labels = []
            self.fermi_energy = self.xml_data.fermi_energy[0]

            #Change rc params
            plt.rcParams['axes.labelsize'] = 2*self.figsize
            plt.rcParams['xtick.bottom'] = False
            plt.rcParams['font.size'] = 2*self.figsize


        def get_highsym_data(self):
            """
            Gets all data relative to the high-symmetry points in the Brillouin
            zone required to perform the plot.
            """
            self.get_kpath_indices()
            self.get_highsym_kpoints()
            self.get_highsym_ticks()
            self.get_highsym_points_labels()

            #Get band gap
            self.get_band_gap()
            self.get_klocs_band_gap()
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
            self.path = np.array([[float(l) for l in line.split()[:3]] for \
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
                    self.labels.append(r"$\Gamma$")
                else:
                    self.labels.append(tuple(p))


        def plot_band_structure(self, save_pdf=True):
            """
            This function plots the band structure
            """

            #Start plot
            fig, ax = plt.subplots(figsize=(self.figsize, self.figsize))
            #Spin polarised case
            if self.xml_data.bs_keywords['lsda'] == 'true':
                ax.plot(
                        self.kpath_idx,
                        self.xml_data.eigvals[:,:\
                                int(self.xml_data.bs_keywords['nbnd_up'])],
                        self.spin_up_colour, label='Spin up',
                        linewidth=self.linewidth,
                        markersize=self.markersize
                        )
                ax.plot(
                        self.kpath_idx,
                        self.xml_data.eigvals[:,\
                                int(self.xml_data.bs_keywords['nbnd_dw']):],
                        self.spin_down_colour,
                        label='Spin down',
                        linewidth=self.linewidth,
                        markersize=self.markersize,
                        alpha=0.4
                        )
            else:
                ax.plot(
                        self.kpath_idx,
                        self.xml_data.eigvals,
                        self.spin_up_colour,
                        linewidth=self.linewidth,
                        markersize=self.markersize
                        )

            #Set energy (y) axis quantities
            ylim = (-self.energy_window / 2 ,\
                    self.energy_window / 2)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(MultipleLocator( self.y_majorticks ))
            ax.yaxis.set_major_formatter(self.y_major_tick_formatter)
            ax.yaxis.set_minor_locator(MultipleLocator( self.y_minorticks ))
            ax.set_ylabel( self.ylabel )

            #Set high-symmetry point quantities
            ax.set_xlim((0.0,1.0))
            ax.set_xticks(self.path_ticks)
            ax.set_xticklabels(
                    self.labels,
                    rotation=45,
                    fontsize=self.figsize * 1.5
                    )

            #Plot vertical lines at each high-symmetry point
            for i, t in enumerate(self.path_ticks):
                ax.axvline(x=t, c='k', linewidth=self.linewidth)

            #Plot horizontal line at the origin (Fermi energy)
            ax.axhline(y=0.0, c='k', linestyle='--', linewidth=self.linewidth)

            #Plot additions for insulating band structure
            if (self.band_gap is not None) and (self.band_gap > 0.0):
                #Coloured in section of gap
                ax.axhspan(self.HO, self.LU, alpha=0.3, color='green')
                #Positions of HO & LU k-points
                for ho_idx in self.kpt_idx_HO:
                    ax.plot(self.kpath_idx[ho_idx], self.HO, 'ko')
                for lu_idx in self.kpt_idx_LU:
                    ax.plot(self.kpath_idx[lu_idx], self.LU, 'ko')

            #Save figure
            if save_pdf:
                fig.tight_layout()
                fig.savefig(self.xml_data.xmlname.replace('xml', 'pdf'))

            return None


        def get_band_gap(self):
            """
            This function computes the band gap of the structure
            """
            #Fixed or smearing case
            if self.xml_data.bands_keywords['occupations'] == 'fixed':
                #Get HO & LU if present
                if ('highestOccupiedLevel' in
                        self.xml_data.bs_keywords.keys()):
                    self.HO = float(self.xml_data.bs_keywords\
                            ['highestOccupiedLevel']) * Hartree
                elif ('lowestUnoccupiedLevel' in
                            self.xml_data.bs_keywords.keys()):
                    self.LU = float(self.xml_data.bs_keywords\
                            ['lowestUnoccupiedLevel']) * Hartree

                #If both HO & LU were found, band gap is the difference
                if (self.HO is not None) and (self.LU is not None):
                    self.band_gap = max(self.LU - self.HO, 0.0)
                #Compute manually if not
                else:
                    self.compute_band_gap()
            #Smearing case
            else:
                self.compute_band_gap()


        def compute_band_gap(self):
            """
            This function computes the band gap explicitly from the k-point
            eigenvalues if necessary.
            """
            #Get highest occupied and lowest unoccupied levels 
            #for spin-polarised and spin-unpolarised cases
            if self.xml_data.bs_keywords['lsda'] == 'true':
                self.HO = max([self.xml_data.eigvals[:,self.nocc - 1].max(),
                    self.xml_data.eigvals[:,self.nocc - 1 +\
                            self.xml_data.nbnd].max()])
                self.LU = min([self.xml_data.eigvals[:,self.nocc].min(),
                    self.xml_data.eigvals[:,self.nocc + self.xml_data.nbnd].\
                            min()])
            else:
                self.HO = self.xml_data.eigvals[:,self.nocc - 1].max()
                self.LU = self.xml_data.eigvals[:,self.nocc].min()

            #Get band gap
            self.band_gap = max(self.LU - self.HO, 0.0)


        def get_klocs_band_gap(self, tol=1e-3):
            """
            Get the indices of the k-points at which the HO & LU occur
            """
            #Get indices of HO k-points
            #Spin-polarised case
            if self.xml_data.bs_keywords['lsda'] == 'true':
                self.kpt_idx_HO = np.append(np.where(
                    abs(self.xml_data.eigvals[:, self.nocc - 1] - self.HO) \
                            < tol
                    ), np.where(abs(
                        self.xml_data.eigvals[:, self.nocc - 1 + \
                                self.xml_data.nbnd]) < tol))
                self.kpt_idx_LU = np.append(np.where(
                    abs(self.xml_data.eigvals[:, self.nocc] - self.LU) < tol
                    ), np.where(abs(
                        self.xml_data.eigvals[:, self.nocc + \
                                self.xml_data.nbnd]) < tol))
            #Spin-unpolarised case
            else:
                self.kpt_idx_HO = np.where(
                    abs(self.xml_data.eigvals[:, self.nocc - 1] - self.HO) < \
                            tol
                    )
                self.kpt_idx_LU = np.where(
                    abs(self.xml_data.eigvals[:, self.nocc] - self.LU) < tol
                    )


def main():
    """
    This main function reads a xml output file generated by PWscf (quantum-
    espresso software package) and outputs the band-structure
    """

    #Get filename and any optional arguments
    xmlname, kwargs = command_line_options()

    #Read XML data file and band structure stuff
    Atoms = XML_Data(xmlname)
    BandStruc = Atoms.band_structure()
    BandStruc.get_highsym_data() #Get high-symmetry points data
    BandStruc.plot_band_structure()

    return None


def command_line_options():
    """
    This function parses the command line options to get the filename.
    """

    #Get filename
    try:
        xmlname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")
    #Other arguments
    argv = sys.argv[2:]

    #Iterate through options
    kwargs = {}
    opts, args = getopt.getopt(argv, "s:w:f:")
    for opt, arg in opts:
        if opt in ['-s', '--save-pdf']:
            if arg in ['true', 'True', 'TRUE']:
                arg = True
            elif arg in ['false', 'False', 'False']:
                arg = False
            kwargs['save_fig'] = arg

        elif opt in ['-w', '--energy-window']:
            kwargs['energy_window'] = float(arg)

        elif opt in ['-f', '--figure-size']:
            kwargs['figsize'] = float(arg)

    return xmlname, kwargs

if __name__ == "__main__":
    main()








