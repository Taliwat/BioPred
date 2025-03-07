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
from rdkit.Chem.rdFingerprintGerator import GetMorganGenerator
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
from scipy.stats import rankdata


# Set file path to read in subject file to start
DATA_DIR = "/home/azureuser/cloudfiles/code/Users/kalpha1865/BioPred/Data/df_files"
df = os.path.join(DATA_DIR, "df_phase_1.parquet")

# Azure ML Setup with the compute cluster
run = Run.get_context()
workspace = Workspace.from_config(path = '/home/azureuser/cloudfiles/code/Users/kalpha1865/BioPred/Config/config.json')
compute_target = ComputeTarget(workspace = workspace, name = 'biopred-cluster-1')




# First encode the Morgan Fingerprints so we can drop the canonical_smiles feature
morgan_gen = GetMorganGenerator(radius = 2, fpSize=2048)

def smiles_to_morgan(smiles):
    """Converts SMILES to a Numpy-compatible Morgan Fingerprint that we can use in modeling."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(morgan_gen.GetFingerprint(mol), dtype=np.uint8) # Converts Native ExplicitBitVect to Numpy

df['morgan_fingerprints'] = df['canonical_smiles'].apply(smiles_to_morgan)


# Encode the molecular_species features, dropping the first value.
df = pd.get_dummies(df, columns = ['molecular_species'], prefix = 'species',
                    drop_first=True, dtype = int)

# Frequency encode both pref_name and target_type
target_type_counts = df['target_type'].value_counts()
df['target_type_freq'] = df['target_type'].map(target_type_counts)

pref_name_counts = df['pref_name'].value_counts()
df['pref_name_freq'] = df['pref_name'].map(pref_name_counts)


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


# Add two more features using scipy's rankdata, using percentile rank.  We saw in our notebooks
# that raw ranking led to too high of a correlation with our target.
df['perc_tank_min_standard_value'] = rankdata(df['min_standard_value']) / len(df)
df['perc_rank_tid'] = df.groupby('tid')['min_standard_value'].transform(lambda x : rankdata(x) / len(x))


# Check skewness and make any corrections for norm dist.
exclude_cols = ['canonical_smiles', 'morgan_fingerprints']

numeric_feats = [col for col in df.select_dtypes(include = np.number).columns if col not in exclude_cols]

for col in numeric_feats:
    skewness = df[col].skew()
    
    if skewness > 1:
        df[col] = np.log1p(df[col])
    elif skewness < -1:
        df[col] = -np.log1p(-df[col])
    else:
        pass


# Continue feature generation with feature aggregation.
exclude_cols = ['canonical_smiles', 'min_standard_value', 'morgan_fingerprints']
agg_feats = [col for col in df.columns if col not in exclude_cols]

# Loop through each current feature and aggregate with our target using the mean.
for feature in agg_feats:
    try:
        agg_mean = df.groupby(feature)['min_standard_value'].mean()
        # Map back to the df
        df[f"{feature}_target_mean"] = df[feature].map(agg_mean)
    except Exception as e:
        print(f"Error with {feature}: {e}")


# More feature generation, this time with pairwise calculation on all of our features generated.
exclude_cols = ['canonical_smiles', 'morgan_fingerprints', 'min_standard_value']

numeric_feats = [col for col in df.columns if col not in exclude_cols]

# Generate all pairwise combinations of new features using itertools
feature_combinations = list(itertools.combinations(numeric_feats, 2))

# Pairwise calculations (with * and /)
for f1, f2 in feature_combinations:
    df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    # Prevent division errors when dividing by 0s, replacing with a small value instead.
    df[f"{f1}_div_{f2}"] = df[f1] / (df[f2].replace(0, 1e-8))


# Now that our feature generation is done let's make sure our data types are consistent.
# Utilize the exclude_cols and numeric_feats above again here.
for col in numeric_feats:
    df[col] = df[col].astype(np.int32)


# Now let's round our features to 3 decimal places using the same logic.
for col in numeric_feats:
    df[col] = np.round(df[col], 3)
# Specifically round min_standard_value, we used the variables for convenience above but we didn't want to do pairwise calculations with our target.
df['min_standard_value'] = np.round(df['min_standard_value'], 3)


# As a sanity check we will introduce a missing value check, and impute with the mean if necessary.
def handle_missing_values(df, cols, strategy = "fill_mean"):
    missing_vals = df[cols].isna().sum()
    total_missing = missing_vals.sum()
    
    if total_missing == 0:
        return df # No changes needed, return the df and move on.
    
    # If missing vals found, proceed with the chosen strategy:
    if strategy == "fill_zero":
        df[cols].fillna(0, inplace = True)
    elif strategy == "fill_mean":
        df[cols].fillna(df[cols].mean(), inplace = True)
    elif strategy == "drop":
        df.dropna(subset = cols, inplace = True)
    
    return df

# Call the function twice now, one for our numeric_feats and the other for our target (doesn't get imputed so different strategy)
df = handle_missing_values(df, numeric_feats, strategy="fill_mean")
df = handle_missing_values(df, ['min_standard_value'], strategy="fill_zero")


# Good now we can move on to examining outliers.
# We will use Z-Score method since we already adjusted our data for normal distribution.
# First though we need to format our data and check for any inconsistencies and possible edge cases before we calculate the z-scores.
# This will help us avoid lots of potential TypeErrors.

for col in numeric_feats:
    # Convert everything to str just for this for loop, and to do these conversions.
    df[col] = df[col].astype(str)
    
    # Replace potential blank values ("" or " ") with 0
    df[col] = df[col].replace(r'^\s*$', "0", regex = True)
    
    # Replace common non-numeric values with a 0
    df[col] = df[col].replace(["None", "N/A", "?", "--", "NaN", "nan"], "0")
    
    # Convert cleaned cols back to numeric, and any remaining errors coerce to 0.
    # Even though 
    df[col] = pd.to_numeric(df[col], errors = 'coerce').fillna(0)

    df[col] = df[col].astype(np.int32)

# Now calculate the zscore with adding percentage of outliers for each feature, will be using a standard 3 for our threshold.
z_score = df[numeric_feats].apply(lambda x: (x - x.mean()) / x.std())

# Detect outliers using the calculated z-score
outliers_z = (np.abs(z_score) > 3).sum()

# Set the outlier percentage, to see what percentage of the data is represented by outliers
outlier_percentage = (outliers_z / len(df)) * 100

# Now as a catch-all, since we need to make sure our data is in check and we won't know how a lot of these new features will perform,
# We will introduce winsorization in case the outlier percentage exceeds the 5%.
for col in numeric_feats:
    if outlier_percentage >= 5:
        lower = df[col].quantile(0.05) # 5th percentile
        upper = df[col].quantile(0.95) # 95th percentile
        df[col] = np.clip(df[col], lower, upper) # Cap values


# Now it is time to move to feature filtration techniques.  We have our generated
# features, and have cleaned and processed them.  Now we will filter them for the best features
# to use in our forthcoming modeling script.  We will use techniques such as RFECV, VIF, and MI Scoring.













