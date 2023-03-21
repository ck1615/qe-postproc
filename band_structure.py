#!/usr/bin/env python3
from structure import XML_Data
import numpy as np
from numpy.linalg import norm
from glob import glob
from misctools import strindex
from ase.units import Hartree, Bohr

#Plotting
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#Parsing command line options
import sys
import getopt


class BandStructure:
    """
    This class contains all the methods and attributes for a band structure
    plot
    """

    def __init__(self, xmlname, figsize=6, ratio=0.8, ymin=-9, ymax=5,
                 units='ase'):
        # self.tolerance:
        self.tol = 1e-10
        # Instantiate outer class XML_Data in the inner class
        self.Structure = XML_Data(xmlname, units=units)
        self.bands_inputfile = self.Structure.xmlname.replace('xml',
                                                              'bands.in')

        # Plot characteristics
        self.ratio = ratio
        self.figsize = figsize
        self.spin_down_colour = ':ro'
        self.spin_up_colour = '--bo'
        self.markersize = self.figsize / 8
        self.linewidth = self.markersize / 3
        self.xlim = (0, 1)
        self.xlabel = 'Wavevectors'
        self.ylabel = r'$E - E_{\mathrm{Fermi}}$ / eV'
        self.y_majorticks = 1.0
        self.y_minorticks = 0.5
        self.y_major_tick_formatter = '{x:.0f}'
        self.ylim = (ymin, ymax)

        # Band gap characteristics
        self.HO = None
        self.LU = None
        self.band_gap = None
        self.nocc = int(np.ceil(float(self.Structure.bs_keywords['nelec'])/2))

        # Bands and path related variables
        self.kpath_idx = []   # Indices from 0.0 to 1.0 of k-points
        self.path = None
        self.path_ticks = None
        self.labels = []
        self.fermi_energy = self.Structure.fermi_energy[0]

        # Change rc params
        plt.rcParams['axes.labelsize'] = 2*self.figsize
        plt.rcParams['xtick.bottom'] = False
        plt.rcParams['font.size'] = 2*self.figsize

        # Get data
        self.get_highsym_data()

    def get_highsym_data(self):
        """
        Gets all data relative to the high-symmetry points in the Brillouin
        zone required to perform the plot.
        """
        self.get_kpath_indices()
        self.get_highsym_kpoints()
        self.get_highsym_ticks()

        # Get band gap
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
        for i, kpt in enumerate(self.Structure.k_points['cartesian']):
            if i == 0:
                path.append(0.0)
            else:
                path.append(path[i-1] + norm(kpt - \
                            self.Structure.k_points['cartesian'][i-1]))

        # Normalise list between 0.0 and 1.0
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
        start_idx = strindex(self.input_lines, "K_POINTS crystal_b")
        end_idx = strindex(self.input_lines, "ATOMIC_POSITIONS")

        #Extract path in crystal coordinates and symbols if present
        self.path = np.array([[float(l) for l in line.split()[:3]] for
            line in self.input_lines[start_idx + 2: end_idx] if \
                    line.split() != []])
        #Get symbols for high-symmetry points
        self.get_highsym_symbols(start_idx, end_idx)

    def get_highsym_symbols(self, start_idx, end_idx):
        """
        Gets the symbols of the high-symmetry points if present
        """

        self.labels = []

        for i, line in enumerate(self.input_lines[start_idx + 2:end_idx]):
            if line.split() == []:
                continue
            last_char = line.split()[-1]
            if '!' in last_char:
                #Modify symbol to make it suitable for plots
                init_symbol = last_char.strip('!')
                symbol = self.get_plottable_symbol(init_symbol)
                self.labels.append(symbol)
            else:
                self.labels.append(tuple(self.path[i]))

    def get_plottable_symbol(self, symbol):
        """
        This function takes the raw capitalised symbol and returns the
        Greek symbol in LaTeX symbols if required.
        """

        bands_input_symbols = {'G': r'$\Gamma$', "G'": r"$\Gamma$'"}
        if symbol in bands_input_symbols:
            return bands_input_symbols[symbol]
        else:
            return symbol

    def get_highsym_ticks(self):
        """
        This function gets the locations of the high-symmetry points along
        the x-axis of the plot.
        """

        #Define high-symmetry point ticks: 
        self.path_ticks = np.zeros(len(self.path))

        #Ensure first and last high-symmetry point correspond with start
        #and end of k-point list
        init_diff = norm(self.path[0] - self.Structure.k_points['crystal'][0]) 
        final_diff = norm(self.path[-1] - self.Structure.k_points['crystal'][-1])

        #print("initial %1.3f and final %1.3f" % (init_diff, final_diff))
        #print(self.path[0],self.Structure.k_points['crystal'][0])
        #print(self.path[-1],self.Structure.k_points['crystal'][-1])

        assert init_diff < self.tol and final_diff < self.tol,\
        "Initial and final are not what is expected:" #+\
        #"initial %1.3f and final %1.3f" % (init_diff, final_diff)

        #Set the values of the first and last ticks
        self.path_ticks[0] = 0.0
        self.path_ticks[-1] = 1.0

        #Initial k-point index
        kpt_idx = 1
        #Iterate over non-extremal high-symmetry points
        for ip, p in enumerate(self.path[1:-1]):
            #Iterate over k-points
            for ik, k in enumerate(self.Structure.k_points['crystal']):
                #Only consider k-points above index
                if ik < kpt_idx:
                    continue
                if norm(k-p) < self.tol:
                    kpt_idx = ik + 1 #Start at next k-point after match
                    self.path_ticks[ip + 1] = self.kpath_idx[ik]
                    break

    def get_band_gap(self):
        """
        This function computes the band gap of the structure
        """
        #Fixed or smearing case
        if self.Structure.bands_keywords['occupations'] == 'fixed':
            #Get HO & LU if present
            if ('highestOccupiedLevel' in
                    self.Structure.bs_keywords.keys()):
                self.HO = float(self.Structure.bs_keywords\
                        ['highestOccupiedLevel']) * Hartree
            elif ('lowestUnoccupiedLevel' in
                        self.Structure.bs_keywords.keys()):
                self.LU = float(self.Structure.bs_keywords\
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
        if self.Structure.bs_keywords['lsda'] == 'true':
            self.HO = max([self.Structure.eigvals[:,self.nocc - 1].max(),
                self.Structure.eigvals[:,self.nocc - 1 +\
                        self.Structure.nbnd].max()])
            try:
                self.LU = min([self.Structure.eigvals[:,self.nocc].min(),
                    self.Structure.eigvals[:,self.nocc + self.Structure.nbnd].\
                            min()])
                self.band_gap = max(self.LU - self.HO, 0.0)
            except IndexError:
                Warning("Cannot compute band gap because not enough bands.")
                self.band_gap = None
                pass

        else:
            self.HO = self.Structure.eigvals[:,self.nocc - 1].max()
            self.LU = self.Structure.eigvals[:,self.nocc].min()
            #Get band gap
            self.band_gap = max(self.LU - self.HO, 0.0)

    def get_klocs_band_gap(self, tol=1e-4):
        """
        Get the indices of the k-points at which the HO & LU occur
        """
        #Get indices of HO k-points
        #Spin-polarised case
        if self.Structure.bs_keywords['lsda'] == 'true':
            self.kpt_idx_HO = np.append(np.where(
                abs(self.Structure.eigvals[:, self.nocc - 1] - self.HO) \
                        < tol
                ), np.where(abs(
                    self.Structure.eigvals[:, self.nocc - 1 + \
                            self.Structure.nbnd]) < tol))
            self.kpt_idx_LU = np.append(np.where(
                abs(self.Structure.eigvals[:, self.nocc] - self.LU) < tol/10
                ), np.where(abs(
                    self.Structure.eigvals[:, self.nocc + \
                            self.Structure.nbnd]) < tol))
        #Spin-unpolarised case
        else:
            self.kpt_idx_HO = np.where(
                abs(self.Structure.eigvals[:, self.nocc - 1] - self.HO) < \
                        tol
                )[0]
            self.kpt_idx_LU = np.where(
                abs(self.Structure.eigvals[:, self.nocc] - self.LU) < tol
                )[0]

    def plot_band_structure(self, save_pdf=True):
        """
        This function plots the band structure
        """

        # Start plot
        fig, ax = plt.subplots(figsize=(self.figsize*self.ratio, self.figsize))
        # Spin polarised case
        if self.Structure.bs_keywords['lsda'] == 'true':
            ax.plot(
                    self.kpath_idx,
                    self.Structure.eigvals[:, :int(self.Structure.
                                           bs_keywords['nbnd_up'])],
                    self.spin_up_colour, label='Spin up',
                    linewidth=self.linewidth,
                    markersize=self.markersize
                    )
            ax.plot(
                    self.kpath_idx,
                    self.Structure.eigvals[:, int(self.Structure.
                                           bs_keywords['nbnd_dw']):],
                    self.spin_down_colour,
                    label='Spin down',
                    linewidth=self.linewidth,
                    markersize=self.markersize,
                    alpha=0.4
                    )
        else:
            ax.plot(
                    self.kpath_idx,
                    self.Structure.eigvals,
                    self.spin_up_colour,
                    linewidth=self.linewidth,
                    markersize=self.markersize
                    )

        # Set energy (y) axis quantities
        ax.set_ylim(self.ylim)
        ax.yaxis.set_major_locator(MultipleLocator(self.y_majorticks))
        ax.yaxis.set_major_formatter(self.y_major_tick_formatter)
        ax.yaxis.set_minor_locator(MultipleLocator(self.y_minorticks))
        ax.set_ylabel(self.ylabel)

        # Set high-symmetry point quantities
        ax.set_xlim((0.0, 1.0))
        ax.set_xticks(self.path_ticks)
        ax.set_xticklabels(
                self.labels,
                rotation=0,
                fontsize=self.figsize * 2
                )

        # Plot vertical lines at each high-symmetry point
        for i, t in enumerate(self.path_ticks):
            ax.axvline(x=t, c='k', linewidth=self.linewidth)

        # Plot horizontal line at the origin (Fermi energy)
        ax.axhline(y=0.0, c='k', linestyle='--', linewidth=self.linewidth)

        # Locations of the LU points
        lu_indices = []

        # Plot additions for insulating band structure
        if (self.band_gap is not None) and (self.band_gap > 0.0):
            # Coloured in section of gap
            ax.axhspan(self.HO, self.LU, alpha=0.3, color='green')
            # Positions of HO & LU k-points
            for ho_idx in self.kpt_idx_HO:
                ax.plot(self.kpath_idx[ho_idx], self.HO, 'ko',
                        ms=self.markersize * 4)

            for lu_idx in self.kpt_idx_LU:
                lu_loc = self.kpath_idx[lu_idx]
                if (lu_loc != 1.0):
                    lu_indices.append(lu_loc)

                ax.plot(lu_loc, self.LU, 'ko', ms=self.markersize*4)

            # If empty sequence
            if len(lu_indices) == 1:
                lu_idx = lu_indices[0]
            else:
                if 0.0 in lu_indices:
                    lu_indices.remove(0.0)
                lu_idx = min(lu_indices)

            # Double arrow indicating band gap
            plt.arrow(
                    lu_idx,
                    self.HO,
                    0.0,
                    self.band_gap,
                    length_includes_head=True,
                    shape='full',
                    color='r',
                    head_width=0.01,
                    head_length=0.1,
                    )
            plt.arrow(
                    lu_idx,
                    self.LU,
                    0.0,
                    -self.band_gap,
                    length_includes_head=True,
                    shape='full',
                    color='r',
                    head_width=0.01,
                    head_length=0.08,
                    )
            # Positions of band gap mention
            x_bg = lu_idx + 0.04
            y_bg = 0.5 * (self.HO + self.LU)
            plt.text(x_bg, y_bg, "{:1.2f} eV".format(self.band_gap),
                     va='center', fontsize=1.6*self.figsize)

        # Save figure
        if save_pdf:
            fig.tight_layout()
            fig.savefig(self.Structure.xmlname.replace('xml', 'pdf'))

        return None


def main():
    """
    This main function reads a xml output file generated by PWscf (quantum-
    espresso software package) and outputs the band-structure
    """

    # Get filename and any optional arguments
    xmlname, kwargs = command_line_options()

    # Read XML data file and band structure stuff
    BS = BandStructure(xmlname, figsize=6, ratio=0.5, ymin=-4, ymax=5)
    BS.plot_band_structure(save_pdf=True)

    return None


def command_line_options():
    """
    This function parses the command line options to get the filename.
    """

    # Get filename
    try:
        xmlname = sys.argv[1]
    except IndexError:
        raise IndexError("No filename has been provided.")
    # Other arguments
    argv = sys.argv[2:]

    # Iterate through options
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
            kwargs['ymin'] = float(arg[1])
            kwargs['ymax'] = float(arg[3])
        elif opt in ['-f', '--figure-size']:
            kwargs['figsize'] = float(arg)

    return xmlname, kwargs

if __name__ == "__main__":
    main()
