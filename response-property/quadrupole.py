import numpy as np


def quadrupole(mf, mol):
    """Kernel of the solver.

    Returns
    -------
    results : dict
        Multipole moment results.
    """
    quadrupole_tensor = mol.intor("int1e_rr")
    dm = mf.make_rdm1()
    norb = mol.nao
    quadrupole_tensor = quadrupole_tensor.reshape((3,3,norb,norb))
    quadrupole_elec = np.einsum('xyij,ji->xy',quadrupole_tensor,dm)

    charges = mol.atom_charges()
    coords = mol.atom_coords()
    print(coords)
    print(coords[0])
    quadrupole_nuc = np.einsum('i,ix,iy->xy',charges,coords,coords)

    tmp = 0.
    for i in range(3):
        tmp += charges[i]*coords[i][0]**2*coords[i][0]**0*coords[i][0]**0
            
    print(-quadrupole_elec[0,0]+tmp)
    quadrupole = -quadrupole_elec + quadrupole_nuc

    return quadrupole
