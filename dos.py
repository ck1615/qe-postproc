#!/usr/bin/env python3
"""
This file contains classes and methods to plot (i) total and (ii) projected
densities of states in spin-polarised and unpolarised cases.
"""

colour_key_file = '/Users/christopherkeegan/.local/bin/qe-parser/colour.key'

#Parsing command line options
import sys
import getopt
import numpy as np
from glob import glob
from misctools import strindex
from ase.units import Hartree, Bohr
import re
from structure import XML_Data

#Plotting
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as colors
import matplotlib.cm as cm


markers = ["x", "+", "d", "1", '*']


class DensityOfStates:
    """
    This class contains all methods and attributes to plot spin-(un)polarised
    electronic densities of states.
    """

    def __init__(self, xmlname, dos_type='projected', figsize=10, ratio=0.6,
            emin=-2, emax=4, units='ase', full_energy_range=False,
            max_nspin=2, savefig=True, angmom=False, total_dos=False, 
            sum_atom_types=True):

        #xml filename and type of DoS desired
        self.xmlname = xmlname
        self.dos_type = dos_type

        #Tolerance for zero
        self.tol = 1e-10
        self.units = units
        self.savefig = savefig
        self.angmom = angmom
        self.sum_atom_types = sum_atom_types
        self.total_dos = total_dos
        self.max_projections = []

        #Total dos filename and array
        self.dos_fname = None
        self.energies = None
        self.dos = None
        self.pdos = None

        #Projected DOS symbols
        self.orbital_number = {'s': 0, 'p': 1, 'd': 2, 'f':3}
        self.angmom_name = {'s': {0: ' '}, 'p': {0: 'z', 1: 'x',\
                2: 'y'}, 'd': {0: "z^{2}", 1: 'xz', 2: 'yz', \
                3: "x^{2} - y^{2}", 4: 'xy'}}


        #Plot characteristics
        self.ratio = ratio
        self.figsize = figsize
        self.spin_down_colour = 'r'
        self.spin_up_colour = 'b'
        self.markersize = 2
        self.linewidth = self.markersize / 3
        self.xlim = (emin, emax)
        self.xlabel = r'$E - E_{\mathrm{Fermi}}$ / eV'
        self.x_majorticks = 2.0
        self.x_minorticks = 1.0
        self.x_major_tick_formatter = '{x:.0f}'
        self.y_major_tick_formatter = '{x:.0f}'
        self.ylabel = 'Density of States'

        #Whether plotting full energy range or not
        self.full_energy_range = full_energy_range

        #XML data
        self.Structure = XML_Data(xmlname, units=self.units)
        #Change rc params
        plt.rcParams['axes.labelsize'] = 2*self.figsize
        plt.rcParams['font.size'] = 2*self.figsize
        plt.rcParams['xtick.major.size'] = 0.7*self.figsize
        plt.rcParams['xtick.minor.size'] = 0.4*self.figsize
        plt.rcParams['ytick.major.size'] = 0.7*self.figsize
        plt.rcParams['ytick.minor.size'] = 0.4*self.figsize

        #Determine spin-polarisation
        self.spin_polarised = None
        self.get_spin_polarisation()
        #Maximum number of spin channels to plot
        if self.spin_polarised:
            self.max_nspin = max_nspin
        else:
            self.max_nspin = 1

        #Get Fermi Energy
        self.get_fermi_energy()


    def read_data_files(self):
        """
        This function extracts the DOS values from the relevant files
        """

        #Read relevant datafiles
        if self.dos_type == 'total':
            self.dos_fname = self.xmlname.replace('xml', 'dos')
            self.read_dos_file()
        elif self.dos_type == 'projected':
            self.dos_fname = self.xmlname.replace('xml', 'pdos_tot')
            self.read_pdos_files()
            self.read_dos_file()
        else:
            raise ValueError("Value of the type of Density of States"+\
                    " plot desired is wrong. Allowed values: total and"+\
                    " projected.")


    def read_pdos_files(self):
        """
        This function reads the projected DOS files
        """
        #Get files:
        pdos_files = glob('{}.pdos_atm*'.format(self.xmlname.strip('.xml')))

        #Initiate pdos data dictionary in spin-polarised and unpolarised cases
        self.pdos = {'up': {}, 'down': {}}

        for fname in pdos_files:
            #Get atom index -- used when not summing over atoms
            atm_idx = int(fname.split('#')[1].split('(')[0])
            #Extract atom, orbital and raw data
            atom, orb, data = self.extract_pdos_data(fname)
            #Allocate data to self.pdos dictionary
            self.allocate_pdos_arrays(atom, orb, data, atm_idx)

        return None


    def allocate_colours(self, atom, orb):
        """
        This function creates a dictionary associating a colours to each orbi-
        -tal of each atom.
        """

        if atom not in self.orbital_colour:
            self.orbital_colour[atom] = {}

        if orb not in self.orbital_colour[atom]:
            self.orbital_colour[atom][orb] = self.colours[self.colour_count]
            self.colour_count += 1


    def extract_pdos_data(self, fname):

        #Extract data and place in dictionary
        #Get locations of parentheses around atom and orbital name in filename
        ls = [int(m.start()+1) for m in re.finditer('\(', fname)]
        rs = [int(m.start()) for m in re.finditer('\)', fname)]
        assert len(ls) == len(rs) == 2, "Number of parentheses not the same"
        atom = fname[ls[0]:rs[0]]
        orb = fname[ls[1]:rs[1]]

        #Check last character of atom name is not a number. If so, remove it
        if atom[-1] in [str(n) for n in range(10)]:
            atom = atom[:-1]

        #Load data
        data = np.loadtxt(fname)
        return atom, orb, data


    def allocate_pdos_arrays(self, atom, orb, data, atm_idx):
        """
        This function takes the atom, orbital and the raw pos data and
        allocates and updates the dictionary of pdos values
        """

        #Allocate energy array (x-axis) if not allocated
        if self.energies is None:
            self.energies = data[:, 0]

        if self.spin_polarised:
            #List indices for pdos values
            up_list = [3 + 2*m for m in range(2 * \
                    self.orbital_number[orb] + 1)]
            dw_list = [4 + 2*m for m in range(2 * \
                    self.orbital_number[orb] + 1)]

            #Summing over atoms
            if self.sum_atom_types:

                #Add atom type to dictionary if not present
                if atom not in self.pdos['up']:
                    for sp in ['up', 'down']:
                        self.pdos[sp][atom] = {}

                if orb not in self.pdos['up'][atom]:
                    self.pdos['up'][atom][orb] = data[:, up_list]
                    self.pdos['down'][atom][orb] = data[:, dw_list]
                else:
                    #Only add to previous array the values of the pDOS(E) and
                    #not the energy array [:,0] as well
                    self.pdos['up'][atom][orb] += data[:, up_list]
                    self.pdos['down'][atom][orb] += data[:, dw_list]

            else:
                #Add particular atom and its index
                if (atom, atm_idx) not in self.pdos['up']:
                    for sp in ['up', 'down']:
                        self.pdos[sp][(atom, atm_idx)] = {}

                self.pdos['up'][(atom, atm_idx)][orb] = data[:, up_list]
                self.pdos['down'][(atom, atm_idx)][orb] = data[:, dw_list]

        else:
            #Index list for pdos values for a given orbital
            idx_list = [2 + m for m in range(2 * \
                        self.orbital_number[orb] + 1)]

            if self.sum_atom_types:
                if atom not in self.pdos['up']:
                    self.pdos['up'][atom] = {}

                if orb not in self.pdos['up'][atom]:
                    self.pdos['up'][atom][orb] = data[:, idx_list]
                else:
                    self.pdos['up'][atom][orb] += data[:, idx_list]

            else:
                if (atom, atm_idx) not in self.pdos['up']:
                    self.pdos['up'][(atom, atm_idx)] = {}

                self.pdos['up'][(atom, atm_idx)][orb] = data[:, idx_list]


    def plot_dos(self):
        """
        This function extracts all the data and plots the  total or projected
        density of states
        """
        #Read the data files
        self.read_data_files()
        #Plot the DOS
        if self.dos_type == 'total':
            self.plot_total_dos()
        elif self.dos_type == 'projected':
            self.plot_projected_dos()

        else:
            raise ValueError("DoS plot type {} not recognized. Allowed "+ \
                    "values are: 'total' and 'projected'.".\
                    format(self.dos_type))


    def get_spin_polarisation(self):

        if (self.Structure.bs_keywords['lsda'] == 'true'):
            self.spin_polarised = True
        else:
            self.spin_polarised = False


    def read_dos_file(self):
        """
        This function reads the prefix.dos file or prefix.pdos_tot file
        in the case of a projected DOS
        """
        try:
            self.dos = np.loadtxt(self.dos_fname)
        except FileNotFoundError:
            raise FileNotFoundError("The total dos file {} was not found".\
                    format(self.dos_fname))


    def get_fermi_energy(self):
        """
        This function extracts the Fermi energy, either from the prefix.dos
        output file if doing a total DOS or from the XML datafile.
        """

        if self.dos_type == 'total':
            with open(self.dos_fname) as f:
                line = f.readline()

            self.fermi_energy = float(line.split()[-2])

        elif self.dos_type == 'projected':
                try:
                    self.fermi_energy = float(self.Structure.bs_keywords\
                            ['two_fermi_energies'].split()[0]) * Hartree
                except KeyError:
                    try:
                        self.fermi_energy = float(self.Structure.bs_keywords\
                            ['fermi_energy'].split()[0]) * Hartree
                    except KeyError:
                        self.fermi_energy = float(self.Structure.bs_keywords\
                            ['highestOccupiedLevel'].split()[0]) * Hartree
        else:
            raise ValueError("{} value not recognised for type of DoS plot."+\
                    "Allowed values are: 'total' and 'projected'.")


    def get_extrema_density_of_states(self):


        idx_min = np.where(self.dos[:,0] > self.fermi_energy + \
                self.xlim[0])[0][0]
        idx_max = np.where(self.dos[:,0] < self.fermi_energy + \
                self.xlim[1])[0][-1]

        ymax = self.dos[idx_min: idx_max + 1, 1].max()
        if self.max_nspin == 2:
            ymin = -self.dos[idx_min: idx_max + 1, 2].max()
        else:
            ymin = 0.0
        self.ylim = (ymin, ymax)


    def plot_projected_dos(self, total_dos=True):
        """
        This function plots the projected Density of States
        """
        #Start figure
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize*self.ratio))

        #List of maximum of each projected DOS channel
        max_projections = []

        #Initialise colour map
        self.num_orbitals = sum(len(val.keys()) for key, val \
                in self.pdos['up'].items())
        cmap = plt.get_cmap('hsv')
        cnorm = colors.Normalize(vmin=0, vmax=self.num_orbitals)
        scalar_map = cm.ScalarMappable(norm=cnorm, cmap=cmap)

        #Determine number of spin-channels to plot
        if self.max_nspin == 2:
            spin_channels = ['up', 'down']
            ax.axhline(y=0.0, c='k', linewidth=1.0)
        else:
            spin_channels = ['up']

        #Iterate over spin-channels
        for i, sp in enumerate(spin_channels):
            #Iterate over atomic species
            for atom in self.pdos[sp]:
                if atom in ['Cu', 'La']:
                    continue
                #Iterate over orbitals

                for orb, pdosvals in self.pdos[sp][atom].items():
                    #Get colour RBG format
                    colour_count=1
                    x = pdosvals[:,0] - self.fermi_energy
                    if self.angmom and (orb != 'f'):
                        #Keep individual orbital angular momentum channels
                        for m in range(2 * self.orbital_number[orb]+1):
                            colour = scalar_map.to_rgba(colour_count)
                            y = (-1)**(i) * pdosvals[:, m + 1]
                            #Append to maximum projectiosn
                            self.get_extrema_density_of_states()
                            #Define labels
                            if orb == "s":
                                label="{} {}".format(atom, orb)
                            else:
                                label = r'{} $\mathrm{{{}_{{{}}}}}$'.\
                                        format(atom, orb , self.angmom_name[orb][m])
                            #Add channel to plot
                            ax.plot(x, y, label=label,
                                    c=colour, linewidth=0.5)
                            #Update colour count
                            colour_count += 1
                    else:
                        #Sum over orbital angular momentum channels 
                        y = (-1) ** (i) * np.sum(pdosvals[:,1:\
                                (2*self.orbital_number[orb]+1+1)], axis=1)
                        max_projections.append(np.max(abs(y)))
                        ax.plot(x, y, label="{} {}".format(atom, orb),
                                c=colour)

            #Add total DOS too in black
            if self.total_dos:
                ax.plot(self.dos[:,0] - self.fermi_energy, (-1)**(i) * \
                    self.dos[:,i+1], 'k', linewidth=0.4, alpha=0.5)

        #Set x-axis quantities
        #Get DOS maximum
        ax.set_xlim(self.xlim)
        ax.set_xlabel( self.xlabel )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        #Set y-axis quantities
        ax.set_ylabel( self.ylabel )
        self.get_extrema_density_of_states()
        ax.set_ylim((-30,30))

        #Add vertical line for Fermi Energy
        ax.axvline(x=0.0, linewidth=1.0, c='k')
        #Finish plot
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),fontsize='xx-small',
                ncol=2)
        plt.tight_layout()
        plt.savefig(self.xmlname.replace('.xml','_pdos.pdf'))


    def plot_total_dos(self, savefig=True):
        """
        This function plots the total electronic density of states.
        """

        #Start plot
        fig, ax = plt.subplots(figsize=(self.figsize, self.figsize*self.ratio))

        #Plot first (or only) spin-channel DoS
        ax.plot(
                self.dos[:,0] - self.fermi_energy,
                self.dos[:,1],
                self.spin_up_colour, label='Spin up',
                linewidth=self.linewidth,
                markersize=self.markersize,
                color='k'
                )
        if self.spin_polarised:
            #Add second spin-channel
            ax.plot(
                    self.dos[:,0] - self.fermi_energy,
                    -self.dos[:,2],
                    self.spin_down_colour,
                    label='Spin down',
                    linewidth=self.linewidth,
                    markersize=self.markersize,
                    color='k'
                    )
            #Add y = 0 line to separate two spin-channels
            ax.axhline(y=0.0, c='k', linewidth = self.linewidth )

        #Add line for Fermi Energy
        ax.axvline(x = 0.0, c='k', linewidth = self.linewidth)

        #Set energy (x) axis quantities
        if self.full_energy_range:
            ax.set_xlim((min(self.dos[:,0]- self.fermi_energy), \
                    max(self.dos[:,0]) - self.fermi_energy))
        else:
            ax.set_xlim( self.xlim )
        ax.xaxis.set_major_locator(MultipleLocator( self.x_majorticks ))
        ax.xaxis.set_minor_locator(MultipleLocator( self.x_minorticks ))
        ax.xaxis.set_major_formatter( self.x_major_tick_formatter )
        ax.set_xlabel( self.xlabel )

        #ax.yaxis.set_major_locator(MultipleLocator( self.y_majorticks ))
        #ax.yaxis.set_minor_locator(MultipleLocator( self.y_minorticks ))
        #ax.yaxis.set_major_formatter( self.y_major_tick_formatter )
        ax.set_ylabel( self.ylabel )

        #Lay figure out properly and save
        plt.tight_layout()
        fig.savefig(self.xmlname.replace("xml", "pdf"))


def main():
    """
    This main function reads a xml output file generated by PWscf (quantum-
    espresso software package) and outputs the band-structure
    """
    #Get filename and any optional arguments
    xmlname, kwargs = command_line_options()

    #Plot DOS
    DOS = DensityOfStates(xmlname, **kwargs)
    DOS.plot_dos()

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
    opts, args = getopt.getopt(argv, "t:f:r:s:m:l:u:")
    for opt, arg in opts:
        if opt in ['-t', '--type-dos-plot']:
            kwargs['dos_type'] = arg
        elif opt in ['-f', '--figure-size']:
            kwargs['figsize'] = float(arg)
        elif opt in ['-r', '--figure-ratio']:
            kwargs['ratio'] = float(arg)
        elif opt in ['-s', '--number-spin-channels']:
            kwargs['max_nspin'] = int(arg)
        elif opt in ['-m', '--m-projection']:
            assert arg in ['0', '1'], "Angular momentum option has to be "+\
                    " either 0 (False) or 1 (True)."
            kwargs['angmom'] = bool(int(arg))
        elif opt in ['-l', '--lower-energy-bound']:
            kwargs['emin'] = float(arg)
        elif opt in ['-u', '--upper-energy-bound']:
            kwargs['emax'] = float(arg)


    return xmlname, kwargs


if __name__ == "__main__":
    main()







