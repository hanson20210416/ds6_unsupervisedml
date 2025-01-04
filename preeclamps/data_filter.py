import pandas as pd
import numpy as np
from proteindata_preprocessing import ProteinDataPreprocessing
from clinicdata_preprocessing import ClinicDataPreprocessing
import matplotlib.pyplot as plt

class FilterData:
    def __init__(self,data):
        self.data = data

    def describe_df(self, df):
        """Generate a summary of DataFrame statistics."""
        summary = pd.DataFrame({
            'Minimum': df.min(),
            'Maximum': df.max(),
            'Median': df.median(),
            'Standard Deviation': df.std(),
            'Ratio of mean with std': df.std() / df.median()
        })
        # Round all values to 2 decimal places
        summary = summary.applymap(lambda x: f"{x:.2f}")
        print(summary)

    def filter_by_hemoglobin(self):
        """Filter data by hemoglobin levels."""
        # Step 1: Load hemoglobin information
        infos = pd.read_excel('/students/2024-2025/master/pre_eclampsy/20241112_MS_pre-eclampsia.xlsx', sheet_name=1)
        hemoglobin_infos = infos[infos['First.Protein.Description'].str.startswith('Hemoglobin', na=False)]['Protein.Ids']
        hemoglobin_list = hemoglobin_infos.tolist()
        # Step 2: Filter and describe the data
        hemoglobin_df = self.data[hemoglobin_list]
        # print(hemoglobin_df)
        self.describe_df(hemoglobin_df)
        # Step 3: Calculate hemoglobin threshold
        if 'P69905' in hemoglobin_df.columns:
            hemoglobin_level = hemoglobin_df['P69905'].median()
            threshold = 0.1 * hemoglobin_level
        else:
            raise KeyError("something wrong.")
        # Step 4: Apply threshold filter
        filtered = (self.data.iloc[:, 1:-1] > threshold).astype(int)  # Exclude the 'PatientCode' and 'group_key' columns
        #print(filtered.head(6))
        # Drop low-expression proteins
        for col in filtered.columns:
            if np.count_nonzero(filtered[col] == 1) / len(filtered) < 1:
                filtered = filtered.drop(columns=[col])
        print(f"Number of remaining columns: {len(filtered.columns)}")
        print(f"The threshold is: {threshold}")
        return self.data[filtered.columns]

    def filter_by_top20_stable_protein(self):
        """Filter top 20 most stable proteins based on standard deviation."""
        top20_index = self.data.iloc[:, 1:-1].std().sort_values(ascending=True).head(20).index.tolist()
        stable_protein_df = self.data[top20_index]
        self.describe_df(stable_protein_df)
        return stable_protein_df

    def filter_by_groups_std(self):
        """Filter data based on group-level standard deviation."""
        grouped = self.data.groupby(['group_key', 'PatientCode']).median()
        medians_df_groups = grouped.groupby(level=0).median()
        # print(medians_df_groups)
        print("Visualizing the standard deviation for each group:")
        std_df = medians_df_groups.std()

        plt.figure(figsize=(6, 4))
        std_df.plot(kind='hist', bins=30, edgecolor='k', alpha=0.7)
        plt.title('Distribution of Standard Deviation Values')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Frequency')
        plt.show()

        threshold = 100000 # Define the threshold by the plot
        filtered_std_df = std_df[std_df > threshold]
        filtered_data = pd.concat([self.data[filtered_std_df.index], self.data['group_key']], axis=1)
        print(f"Filtered data based on standard deviation threshold: {threshold}")
        print(f' The number of columns after filtering: {len(filtered_data.columns)}')
        return filtered_data



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
    #print(group_data)
    uniqueid_df = data_processor.add_tag(unique_df, group_data)
    print(uniqueid_df.head())
    # Filter data   
    filter_data_homoglobin = FilterData(uniqueid_df).filter_by_hemoglobin()
    print(filter_data_homoglobin.head())
    filter_data_std = FilterData(uniqueid_df).filter_by_groups_std()
    print(filter_data_std.head())
    # filter_data_by_stable_protein = FilterData(uniqueid_df).filter_by_top20_stable_protein()
    # print(filter_data_by_stable_protein)
