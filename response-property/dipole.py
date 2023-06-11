import numpy as np

def dipole(mf, mol):
    """Kernel of the solver.

    Returns
    -------
    results : dict
        Multipole moment results.
    """
    dipole_tensor = mol.intor("int1e_r")
    dm = mf.make_rdm1()
    dipole_elec = np.einsum('xij,ij->x',dipole_tensor,dm)

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    dipole_nuc = np.einsum('i,ix->x',charges,coords)

    dipole = -dipole_elec + dipole_nuc

    return dipole


