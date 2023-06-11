from pyscf import gto, scf
from dipole import dipole
from quadrupole import quadrupole
from polar import polar
import numpy as np

mol = gto.M(atom='O 0.0 -0.07579405737426999 0.0;H 0.8668135009519599 0.60143111057846 0; H -0.8668135009519599 0.60143111057846 0.0', basis='sto-3g')
mf = scf.RHF(mol)
mf.kernel()

#a = dipole(mf, mol)
#b = quadrupole(mf, mol)
c = polar(mf, mol)
#print(a)
#print(a/0.393456)
#print(b)
#print(b/0.393456*0.529177)
print(c)
