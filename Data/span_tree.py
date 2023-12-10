import csv
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def delete_h(mol):
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atom.SetNoImplicit(True)



def get_rings(mol):
    rings = mol.GetRingInfo()
    r = rings.AtomRings()
    r_list = []
    for i in range(len(r)):
        r_list.append(list(r[i]))
    return r_list


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def isRingAromatic(mol,ring):

    if False in [mol.GetBondWithIdx(index).GetIsAromatic()  for index in ring]:
        return False
    else:
        return True


def get_ar_rings(mol):

    ar_rings = []
    rings = mol.GetRingInfo()
    if not rings:
        return False
    else:
        bond_rings = rings.BondRings()
        for i in range(len(bond_rings)):
            if isRingAromatic(mol, bond_rings[i]):
                ar_rings.append(list(rings.AtomRings()[i]))
        return ar_rings



def spanning_tree(A, Aromatic_rings, r_list, COO, CO, C3C):
    drop_list = []


    for ring in r_list:
        if ring not in Aromatic_rings:
            A[ring[0], ring[1]] = 0
            A[ring[1], ring[0]] = 0


    for i in range(len(Aromatic_rings)):

        a = Aromatic_rings[i][0]
        k = 0
        for j in range(len(Aromatic_rings[i])-1):
            if a not in drop_list:
                b = Aromatic_rings[i][j+1]
                if b not in drop_list:
                    A[a, :] += A[b, :]
                    A[:, a] += A[:, b]
                    drop_list.append(b)
                else:
                    continue
            else:
                k += 1
                a = Aromatic_rings[i][k]
                if k == len(Aromatic_rings[i]):
                    break


    for i in range(len(COO)):
        for j in range(len(COO[i])-1):
            if COO[i][j+1] not in drop_list:
                A[COO[i][0], :] += A[COO[i][j+1], :]
                A[:, COO[i][0]] += A[:, COO[i][j+1]]
                drop_list.append(COO[i][j+1])
            else:
                continue


    for i in range(len(CO)):
        for j in range(len(CO[i]) - 1):
            if CO[i][j + 1] not in drop_list:
                A[CO[i][0], :] += A[CO[i][j + 1], :]
                A[:, CO[i][0]] += A[:, CO[i][j + 1]]
                drop_list.append(CO[i][j + 1])
            else:
                continue


    for i in range(len(C3C)):
        for j in range(len(C3C[i])-1):
            if C3C[i][j+1] not in drop_list:
                A[C3C[i][0], :] += A[C3C[i][j+1], :]
                A[:, C3C[i][0]] += A[:, C3C[i][j+1]]
                drop_list.append(C3C[i][j+1])
            else:
                continue


    drop_list = sorted(drop_list, reverse=True)
    for i in drop_list:
        A = np.delete(A, i, axis=0)
        A = np.delete(A, i, axis=1)


    for i in range(len(A)):
        for j in range(len(A)):
            A[i][i] = 0
            if A[i][j] >= 1:
                A[i][j] = 1

    return A



dataset = 'bace'

with open('./Data/molecules/' + dataset + '.csv') as f:
    with open('./Data/trees/' + dataset + '_tree.csv', 'w') as file:
        f_csv = csv.reader(f)
        next(f_csv)
        for index, row in enumerate(f_csv):
            smile = row[0]
            mol = Chem.MolFromSmiles(smile)
            delete_h(mol)
            mol = mol_with_atom_index(mol)


            patt = []
            patt.append(Chem.MolFromSmarts('[C](-[OH])=[O]'))
            patt.append(Chem.MolFromSmarts('[C](=[O])-[OH]'))
            patt.append(Chem.MolFromSmarts('[N+](-[O-])=[O]'))
            patt.append(Chem.MolFromSmarts('[N+](=[O])-[O-]'))
            # patt.append(Chem.MolFromSmarts('[C](=[O])'))

            rep = Chem.MolFromSmarts('C')

            for i in range(len(patt)):
                while (True):
                    if mol.HasSubstructMatch(patt[i]):
                        m = AllChem.ReplaceSubstructs(mol, patt[i], rep)
                        mol = m[0]
                    else:
                        break

            smile = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smile)


            print('molecule', index)
            patt = Chem.MolFromSmarts('[C](=[O])-[O]')
            COO = []
            flag = mol.HasSubstructMatch(patt)
            if flag:
                COO = mol.GetSubstructMatches(patt)
                print("COO:", COO)

            patt = Chem.MolFromSmarts('[C](=[O])')
            CO = []
            flag = mol.HasSubstructMatch(patt)
            if flag:
                CO = mol.GetSubstructMatches(patt)
                print("CO:", CO)

            patt = Chem.MolFromSmarts('C#C')
            C3C = []
            flag = mol.HasSubstructMatch(patt)
            if flag:
                C3C = mol.GetSubstructMatches(patt)
                print("C3C:", C3C)


            Aromatic_rings = get_ar_rings(mol)
            r_list = get_rings(mol)

            A = Chem.rdmolops.GetAdjacencyMatrix(mol)
            A = spanning_tree(A, Aromatic_rings, r_list, COO, CO, C3C)

            G = nx.from_numpy_matrix(A)
            cycle = nx.cycle_basis(G)
            if len(cycle) is not 0:

                for i in range(len(cycle)):
                    A[cycle[i][0], cycle[i][1]] = 0
                    A[cycle[i][1], cycle[i][0]] = 0

            myWriter = csv.writer(file)

            myWriter.writerow([index+1, len(A)])
            myWriter.writerows(A)