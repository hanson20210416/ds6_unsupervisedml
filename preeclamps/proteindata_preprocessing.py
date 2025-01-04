import pandas as pd
import numpy as np
from scipy import stats
from clinicdata_preprocessing import ClinicDataPreprocessing

class ProteinDataPreprocessing():
    def __init__(self, protein_file):
        self.protein_file = protein_file

    def load_protein_data(self):
        """Load protein data from the provided Excel file"""
        protein = pd.read_excel(self.protein_file, sheet_name=1)
        protein = protein.T
        df = protein.drop(['Protein.Group', 'Protein.Names', 'First.Protein.Description', 'Genes'], axis=0)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=False)
        df = df.iloc[:, :-3]  # drop the summary rows
        df = df.rename(columns={'index': 'PatientCode'})
        return df
    
    def delete_unrelevant_features(self, df):
        """Delete columns with more than 30% missing data and columns starting with 'contaminant_'"""
        missing_ratio = df.isna().sum() / len(df)
        relevant_columns = [column for column, ratio in missing_ratio.items() if ratio < 0.3 and not column.startswith('contaminant_')]
        return df[relevant_columns]
    
    def rename_rows(self, df):
        """Rename rows based on splitting 'PatientCode' column to align with the clinical data"""
        rename_dict = {}
        for i in df.iloc[:, 0]:
            parts = i.split('_')
            if len(parts) >= 3:
                name = parts[2]
                rename_dict[i] = name
            else:
                print(f"Skipping invalid name: {i}")
        df_renamed = df.copy()  # Create a copy of the DataFrame
        df_renamed.iloc[:, 0] = df_renamed.iloc[:, 0].map(rename_dict)
        return df_renamed

    
    def check_features_distribution(self, df_renamed):
        """Check if the columns follow a normal distribution using the Shapiro-Wilk test"""
        normal_distributed = []
        non_normal_distributed = []
        for column in df_renamed.columns[1:]:
            stat, p = stats.shapiro(df_renamed[column].dropna())
            #print(f"{column}: Statistics={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                normal_distributed.append(column)
            else:
                non_normal_distributed.append(column)
        print(f"there are {len(normal_distributed)} columns that looks normally distributed (p > 0.05)")
        print(f"there are {len(non_normal_distributed)} columns that does not look normally distributed (p <= 0.05)")

    def impute_missing_values(self, df_renamed, method='median'):
        """Impute missing values for samples using 'mean' or 'median'"""
        for sample in df_renamed['PatientCode'].unique():
            sample_same = df_renamed[df_renamed['PatientCode'] == sample]
            for col in df_renamed.columns[1:]:
                if sample_same[col].isna().any():
                    if sample_same[col].notna().sum() >= 1:
                        fill_value = sample_same[col].dropna().iloc[0]
                        # Use .loc to safely set values
                        df_renamed.loc[df_renamed['PatientCode'] == sample, col] = sample_same[col].fillna(fill_value)
                    elif sample_same[col].isna().all():
                        # Ensure you're modifying the original DataFrame
                        fill_value = df_renamed[col].median() if method == 'median' else df_renamed[col].mean()
                        df_renamed.loc[df_renamed['PatientCode'] == sample, col] = fill_value
        print("Missing values imputed successfully!")
        # print(df_renamed.head())
        return df_renamed

    def unique_samples(self, imputed_df):
        """Group by samples and calculate the mean for each group"""
        uniqueid_df = imputed_df.groupby('PatientCode', as_index=False).mean()
        return uniqueid_df.dropna(axis=1, how='all')

    def add_tag(self, uniqueid_df, group_data):
        """Add tags based on group patient codes"""
        # Ensure the 'group_key' column exists in uniqueid_df
        if 'group_key' not in uniqueid_df.columns:
            uniqueid_df['group_key'] = pd.NA  # Initialize with missing values
        # Iterate over group_data and tag matching rows
        for group, ids in group_data.items():
            # Ensure all IDs are uppercase strings
            ids = [str(id).upper() for id in ids if pd.notna(id)]
            # Tag rows in uniqueid_df where 'PatientCode' matches
            uniqueid_df.loc[uniqueid_df['PatientCode'].isin(ids), 'group_key'] = group
        return uniqueid_df


if __name__ == "__main__":
    clinic_file = '/students/2024-2025/master/pre_eclampsy/pre-eclampsia.xlsx'
    protein_file = '/students/2024-2025/master/pre_eclampsy/20241112_MS_pre-eclampsia.xlsx'
    data_processor = ProteinDataPreprocessing(protein_file)
    df = data_processor.load_protein_data()
    df1 = data_processor.delete_unrelevant_features(df)
    df_renamed = data_processor.rename_rows(df1)
    data_processor.check_features_distribution(df_renamed)
    df_imputed = data_processor.impute_missing_values(df_renamed)
    unique_df = data_processor.unique_samples(df_imputed)
    # Clinic Data Processing
    clinic_processor = ClinicDataPreprocessing()
    group1, group2, group3, group4, group5, group6 = clinic_processor.load_data(clinic_file)
    
    # Prepare group data for tagging
    group_data = {}
    for i, group in enumerate([group1, group2, group3, group4, group5, group6], start=1):
        if 'Patiëntcode' in group.columns:
            group['Patiëntcode'].replace('NaN', pd.NA, inplace=True)
            group.dropna(subset=['Patiëntcode'], inplace=True)
            group_data[f'group{i}'] = group['Patiëntcode'].astype(str).str.upper().tolist()
        else:
            print(f"Warning: 'Patiëntcode' column not found in group{i}")
    print(group_data)
    uniqueid_df = data_processor.add_tag(unique_df, group_data)
    print(uniqueid_df.head())

