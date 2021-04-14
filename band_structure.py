"""
Classes and methods to plot band structures
"""
from structure import XML_Data
from numpy.linalg import norm
import matplotlib.pyplot as plt


class Bands:
    """
    Class taking .xml file data and, if 'bands' calculation was used, generates
    band structure plots
    """

    def __init__(self, xmlname):

        #Output file data
        self.xml_data = XML_Data(xmlname)

        #Band structure variables
        self.kpath_idx = None


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
        self.kpath_idx = [(idx - path[0]) / (path[-1] - \
                path[0]) for idx in path]

    def shift_eigenvalues(self):
        """
        This function shifts the eigenvalues such that the Fermi energy
        corresponds to 0.
        """
        self.xml_data.get_fermi_energy()
        self.shifted_eigvals = self.xml_data.eigvals - self.xml_data.fermi_energy[0]
