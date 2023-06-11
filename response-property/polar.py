import numpy as np
import pyscf 

def polar(mf, mol):
    """Kernel of the solver.

    Returns
    -------
    results : dict
        Multipole moment results.
    """
    Fao = mf.get_fock()
    eri_ao = mol.intor('int2e')


    C = mf.mo_coeff
    Fmo = np.dot(C.T, Fao).dot(C)
    # get the two-electron integrals as a numpy array
    eri_mo = np.einsum('pqrs,pt->tqrs', eri_ao, C, optimize=True)
    eri_mo = np.einsum('pqrs,qt->ptrs', eri_mo, C, optimize=True)
    eri_mo = np.einsum('pqrs,rt->pqts', eri_mo, C, optimize=True)
    eri_mo = np.einsum('pqrs,st->pqrt', eri_mo, C, optimize=True)

    A = np.einsum('ij,ab->iajb', np.eye(Fmo.shape[0]), Fmo, optimize=True)
    A -= np.einsum('ab,ij->iajb', np.eye(Fmo.shape[0]), Fmo, optimize=True)
    A += 2 * np.einsum('aijb->iajb', eri_mo, optimize=True)
    A -= np.einsum('jiab->iajb', eri_mo, optimize=True)

    B = 2 * np.einsum('aijb->iajb', eri_mo, optimize=True)
    B -= np.einsum('ajbi->iajb', eri_mo, optimize=True)

    H = A + B

    norb = mol.nao
    ndocc = mol.nelec[0]
    nvirt = norb - ndocc
    nrot = ndocc * nvirt


    H = H[:ndocc, ndocc:, :ndocc, ndocc:]
    H = H.reshape(nrot, nrot)

    dipole_tensor = -2*mol.intor("int1e_r")
    dipole_list = []
    for i in range(len(dipole_tensor)):
        dipole_tensor[i] = np.dot(C.T, dipole_tensor[i]).dot(C)
        dipole_list.append(dipole_tensor[i][:ndocc, ndocc:].ravel())

    responses = []
    for perturbation in dipole_list:
        responses.append(np.linalg.solve(H, -perturbation))

    polarizabilities = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            polarizabilities[i, j] = -np.dot(dipole_list[i], responses[j])

    return polarizabilities
