# List libraries needed for this notebook.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import azureml.core
from azureml.core import Workspace, Experiment, Dataset, ComputeTarget
from azureml.core.run import Run
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, GraphDescriptors, AllChem
from rdkit.DataStructs import BitVectToText
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set file path to read in subject file to start
DATA_DIR = "/home/azureuser/cloudfiles/code/Users/kalpha1865/BioPred/Data/df_files"
df = os.path.join(DATA_DIR, "df_phase_1.parquet")

# Azure ML Setup with the compute cluster
run = Run.get_context()
workspace = Workspace.from_config(path = '/home/azureuser/cloudfiles/code/Users/kalpha1865/BioPred/Config/config.json')
compute_target = ComputeTarget(workspace = workspace, name = 'biopred-cluster-1')

# First encode the Morgan Fingerprints so we can drop the canonical_smiles feature
def smiles_to_morgan(smiles, radius = 2, n_bits = 2048):
    """Converts SMILES to a Numpy-compatible Morgan Fingerprint that we can use in modeling."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = n_bits)
    return np.array(fp)

df['morgan_fingerprints'] = df['canonical_smiles'].apply(smiles_to_morgan)

# Encode the molecular_species and activity_count features, dropping the first value.
df = pd.get_dummies(df, columns = ['molecular_species'], prefix = 'species',
                    drop_first=True, dtype = int)

# Start feature generation now using RDKit, again using canonical_smiles to generate.
def rdkit_features(smiles):
    """Uses the RDKit library to select and generate new features from our SMILES data."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "bertz_ct" : GraphDescriptors.BertzCT(mol),
        "balaban_j" : Descriptors.BalabanJ(mol),
        "tpsa" : Descriptors.TPSA(mol),
        "fraction_csp3" : Descriptors.FractionCSP3(mol),
        "num_rings" : Descriptors.RingCount(mol),
        "labute_asa" : Descriptors.LabuteASA(mol),
        "molecular_volume" : Descriptors.MolMR(mol)
    }

df['rdkit_feats'] = df['canonical_smiles'].apply(rdkit_features)
rdkit_feats_df = pd.DataFrame(df['rdkit_feats'].tolist())
df = pd.concat([df, rdkit_feats_df], axis = 1)
df.drop(columns = ['rdkits_feats'], inplace = True)

# Continue feature generation with feature aggregation.
exclude_cols = ['canonical_smiles', 'min_standard_value', 'morgan_fingerprints', ]














