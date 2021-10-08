#!/usr/bin/env python3
"""
This set of functions is used to plot the electronic band structure combined
with the density of states projected onto atomic orbitals on particular sites
in each of the two Cu-O layers of La2CuO4
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
import matplotlib.cm as cm
from dos import DensityOfStates as dos
import sys
import getopt


def get_atom_indices(calculation):

    if calculation == "lesco":
        atom_indices = {'Cu': {
                'LTT': {1: [1, 4], 2: [2, 3]},
                'LTO': {1: [1, 2], 2: [3, 4]}
            },

                            'O(ax)': {
                'LTT': {1: [13, 16, 17, 20], 2: [18, 19, 14, 15]},
                'LTO': {1: [17, 18, 19, 20], 2: [13, 14, 15, 16]}
            },
                            'O(eq)': {
                'LTT': {1: [22, 24, 25, 27], 2: [21, 23, 26, 28]},
                'LTO': {1: [21, 22, 23, 24], 2: [25, 26, 27, 28]}
            }
                         }
    elif calculation == "CuU_9":
        atom_indices = {'Cu': {
                'LTT': {1: [1, 4], 2: [2, 3]},
                'LTO': {1: [9, 12], 2: [10, 11]}
            },

                            'O(ax)': {
                'LTT': {1: [13, 16, 17, 20], 2: [14, 15, 18, 19]},
                'LTO': {1: [13, 16, 17, 20], 2: [14, 15, 18, 19]}
            },
                            'O(eq)': {
                'LTT': {1: [21, 23, 25, 27], 2: [22, 24, 26, 28]},
                'LTO': {1: [21, 23, 25, 27], 2: [22, 24, 26, 28]},
            }
                         }
    else:
        return ValueError("calculation name must be 'lesco' or 'CuU_9'")

    return atom_indices


def get_data(fname, element, calculation, xlim=(-8, 4), angmom=True,
             sum_atom_types=False, ylim=(-10, 10)):

    layers = [1, 2]
    Struc = dos(fname, angmom=angmom, sum_atom_types=sum_atom_types)
    phase = fname.split("_")[-1].split('.')[0]

    spread = xlim[1] - xlim[0]

    # Name
    if calculation == 'lesco':
        plotname = "LESCO"
    elif calculation == 'CuU_9':
        plotname == "La2CuO4_9.4"

    majorloc = round(spread / 10, 1)
    if majorloc >= 1:
        majorloc = round(majorloc, 0)
    minorloc = majorloc / 2

    indices = get_atom_indices(calculation)[element][phase]

    # Read data files
    Struc.read_data_files()

    # Plot for each layer
    for layer in layers:

        # Atom dependent stuff
        if element == "Cu":
            elem = element
            atom = r'{}$^{{({})}}$'.format(element, layer)
            orbitals = ['s', 'p', 'd']
            num_orbitals = 9

        elif element in ['O(ax)', 'O(eq)']:
            elem = 'O'
            atom = r'{}$^{{({})}}$'.format(element, layer)
            orbitals = ['s', 'p']
            num_orbitals = 4

        # Get data
        data = {sp: {
            orb: sum(Struc.pdos[sp][(elem, i)][orb] for i in indices[layer])
            for orb in orbitals
        }
                for sp in ['up', 'down']
               }

        fig, ax = plt.subplots(figsize=(Struc.figsize,
                               Struc.figsize*Struc.ratio))

        if Struc.max_nspin == 2:
            spin_channels = ['up', 'down']
            ax.axhline(y=0.0, c='k', linewidth=0.0)

        else:
            spin_channels = ['up']

        # Colour map
        cmap = plt.get_cmap('hsv')
        cnorm = colors.Normalize(vmin=0, vmax=num_orbitals)
        scalar_map = cm.ScalarMappable(norm=cnorm, cmap=cmap)

        for i, sp in enumerate(spin_channels):
            colour_count = 1
            for orb in orbitals:
                pdosvals = data[sp][orb]
                x = Struc.energies - Struc.fermi_energy

                for m in range(2 * Struc.orbital_number[orb] + 1):
                    colour = scalar_map.to_rgba(colour_count)
                    y = (-1) ** (i) * pdosvals[:, m]

                    if orb == "s":
                        label = "{} {}".format(atom, orb)
                    else:
                        label = r'{} $\mathrm{{{}_{{{}}}}}$'.\
                            format(atom, orb, Struc.angmom_name[orb][m])

                    ax.plot(x, y, label=label, c=colour)
                    colour_count += 1

        ax.set_xlim(xlim)
        ax.set_xlabel(Struc.xlabel)
        ax.xaxis.set_major_locator(MultipleLocator(majorloc))
        ax.xaxis.set_minor_locator(MultipleLocator(minorloc))

        ax.set_ylabel(Struc.ylabel)
        ax.set_ylim(ylim)
        ax.axvline(x=0.0, linewidth=1.0, c='k')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='xx-small',
                   ncol=2)
        plt.tight_layout()
        plt.savefig('{}_{}_Rotated_pdos_{}_layer{}.pdf'.format(plotname, phase,
                    element, layer))


def main():
    """Get filename"""

    # Get filename and any optional arguments
    xmlname, element, calculation, kwargs = command_line_options()

    # Plot
    get_data(xmlname, element, calculation, **kwargs)

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

    try:
        element = sys.argv[2]
    except IndexError:
        raise IndexError("No element has been provided.")
    try:
        calculation = sys.argv[3]
    except IndexError:
        raise IndexError("No element has been provided.")

    # Other arguments
    argv = sys.argv[4:]

    # Iterate through options
    kwargs = {}
    opts, args = getopt.getopt(argv, "s:w:f:")
    for opt, arg in opts:
        if opt in ['-w', '--energy-window']:
            kwargs['ylim'] = (float(arg[1]), float(arg[3]))
        elif opt in ['-d', '--dos-window']:
            kwargs['xlim'] = (float(arg[1]), float(arg[3]))

    return xmlname, element, calculation, kwargs


if __name__ == "__main__":
    main()
