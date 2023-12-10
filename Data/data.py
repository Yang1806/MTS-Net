"""
    File to load dataset based on user control from main file
"""
from Data.molecule  import MoleculeDataset

def LoadData(DATASET_NAME, seed):
    return MoleculeDataset(DATASET_NAME, seed)
