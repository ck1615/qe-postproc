"""
This file contains classes and methods to plot (i) total and (ii) projected
densities of states in spin-polarised and unpolarised cases.
"""

from structure import XML_Data
import numpy as np
from glob import glob
from misctools import strindex
from ase.units import Hartree, Bohr

#Plotting
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#Parsing command line options
import sys
import getopt

class DensityOfStates:
    """
    This class contains all methods and attributes to plot spin-(un)polarised
    electronic densities of states.
    """

    def __init__(self, xmlname, dos_type='total', figsize=10, ratio=0.6,
            emin=-10, emax=10, units='ase', total_erange=False):

        #xml filename and type of DoS desired
        self.xmlname = xmlname
        self.dos_type = dos_type

        #Tolerance for zero
        self.tol = 1e-10
        self.units = units

        #XML data
        self.Structure = XML_Data(xmlname, units=self.units)
        #Determine spin-polarisation
        self.spin_polarisation = None
        self.get_spin_polarisation()

        #Total dos filename and array
        self.dos_fname = self.xmlname.replace('xml','dos')
        self.dos = None

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
        self.y_majorticks = 25
        self.y_minorticks = 12.5
        self.y_major_tick_formatter = '{x:.0f}'
        self.ylabel = 'Density of States / a.u.'

        #Whether plotting full energy range or not
        self.total_erange = total_erange


        #Change rc params
        plt.rcParams['axes.labelsize'] = 2*self.figsize
        plt.rcParams['font.size'] = 2*self.figsize
        plt.rcParams['xtick.major.size'] = 0.7*self.figsize
        plt.rcParams['xtick.minor.size'] = 0.4*self.figsize
        plt.rcParams['ytick.major.size'] = 0.7*self.figsize
        plt.rcParams['ytick.minor.size'] = 0.4*self.figsize


    def plot_dos(self, savefig=True):
        """
        This function extracts all the data and plots the  total or projected
        density of states
        """

        #Spin-polarisation
        self.get_spin_polarisation()
        #Fermi energy
        self.get_fermi_energy()

        if self.dos_type == 'total':
            #Read prefix.dos file
            self.read_dos_file()
            #Normalise the DOS
            self.normalise_total_dos()
            #Plot DOS
            self.plot_total_dos()
        else:
            raise NotImplementedError


    def get_spin_polarisation(self):

        if self.Structure.bs_keywords['lsda'] == 'true':
            self.spin_polarised = True
        else:
            self.spin_polarised = False


    def read_dos_file(self):
        """
        This function reads the prefix.dos file
        """
        if self.spin_polarised is None:
            self.get_spin_polarisation()

        try:
            self.dos = np.loadtxt(self.dos_fname)
        except FileNotFoundError:
            raise FileNotFoundError("The total dos file {} was not found".\
                    format(dos_fname))

    def normalise_total_dos(self):
        """
        This function normalises the total DOS between 0 and 100
        """

        norm_const = 100 / max(self.dos[:,1])
        self.dos[:,1] *= norm_const
        self.dos[:,2] *= norm_const


    def get_fermi_energy(self):
        """
        This function extracts the Fermi energy, either from the prefix.dos
        output file if doing a total DOS or from the XML datafile.
        """

        if self.dos_type == 'total':
            with open(self.dos_fname) as f:
                line = f.readline()

            self.fermi_energy = float(line.split()[-2])

        elif self.dos_type == 'pdos':
            if self.spin_polarised:
                try:
                    self.fermi_energy = float(self.Structure.bs_keywords\
                            ['two_fermi_energies'].split()[-1]) * Hartree
                except KeyError:
                    try:
                        self.fermi_energy = float(self.Structure.bs_keywords\
                            ['fermi_energy'].split()[-1]) * Hartree
                    except KeyError:
                        self.fermi_energy = float(self.Structure.bs_keywords\
                            ['highestOccupiedLevel'].split()[-1]) * Hartree
            else:
                try:
                    self.fermi_energy = float(self.Structure.bs_keywords\
                       ['fermi_energy'].split()[-1]) * Hartree
                except KeyError:
                    self.fermi_energy = float(self.Structure.bs_keywords\
                       ['highestOccupiedLevel'].split()[-1]) * Hartree


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
                markersize=self.markersize
                )
        if self.spin_polarised:
            #Add second spin-channel
            ax.plot(
                    self.dos[:,0] - self.fermi_energy,
                    -self.dos[:,2],
                    self.spin_down_colour,
                    label='Spin down',
                    linewidth=self.linewidth,
                    markersize=self.markersize
                    )
            #Add y = 0 line to separate two spin-channels
            ax.axhline(y=0.0, c='k', linewidth = self.linewidth )

        #Add line for Fermi Energy
        ax.axvline(x = 0.0, linewidth = self.linewidth, c='k')

        #Set energy (x) axis quantities
        if self.total_erange:
            ax.set_xlim((min(self.dos[:,0]- self.fermi_energy), \
                    max(self.dos[:,0]) - self.fermi_energy))
        else:
            ax.set_xlim( self.xlim )
        ax.xaxis.set_major_locator(MultipleLocator( self.x_majorticks ))
        ax.xaxis.set_minor_locator(MultipleLocator( self.x_minorticks ))
        ax.xaxis.set_major_formatter( self.x_major_tick_formatter )
        ax.set_xlabel( self.xlabel )

        #Set DOS (y) axis quantities
        if self.spin_polarised:
            ax.set_ylim((-round((max(self.dos[:,2]) + 10)/10)*10, 110))
        else:
            ax.set_ylim((0, 110))

        ax.yaxis.set_major_locator(MultipleLocator( self.y_majorticks ))
        ax.yaxis.set_minor_locator(MultipleLocator( self.y_minorticks ))
        ax.yaxis.set_major_formatter( self.y_major_tick_formatter )
        ax.set_ylabel( self.ylabel )


def main():
    """
    This main function reads a xml output file generated by PWscf (quantum-
    espresso software package) and outputs the band-structure
    """

    #Get filename and any optional arguments
    xmlname, kwargs = command_line_options()

    #Plot DOS
    DOS = DensityOfStates(xmlname, kwargs)
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
    opts, args = getopt.getopt(argv, "t:f:r:")
    for opt, arg in opts:
        if opt in ['-t', '--save-pdf']:
            kwargs['dos_type'] = arg
        elif opt in ['-f', '--figure-size']:
            kwargs['figsize'] = float(arg)
        elif opt in ['-r', '--figure-ratio']:
            kwargs['ratio'] = float(arg)

    return xmlname, kwargs


if __name__ == "__main__":
    main()







