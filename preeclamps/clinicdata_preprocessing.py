import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt

import pandas as pd

class ClinicDataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, clinic_file):
        """Load clinical data from an Excel file"""
        data = pd.read_excel(clinic_file, sheet_name=[0, 1, 2, 3, 4, 5])
        group1 = data[0]
        group2 = data[1]
        group3 = data[2]
        group4 = data[3]
        group5 = data[4]
        group6 = data[5]
        return group1, group2, group3, group4, group5, group6
    

    def find_locations_missing_values_group1(self, group):
        """Find locations of missing values in the given group1"""
        group['Patiëntcode'].replace('NaN', pd.NA, inplace=True)
        group = group.dropna(subset=['Patiëntcode'])
        print(f"Missing values count for group: {group.isna().sum().sum()}")
        # Display the locations of missing values
        missing_values_location = group.isna()
        missing_values_positions = missing_values_location.stack()[missing_values_location.stack()]
        print(f"Locations of missing values:\n{missing_values_positions}")
        return group
    
    def find_locations_missing_values_groups(self, group):
        """Find locations of missing values in the given group"""
        print(f"Missing values count for groups:")
        print(group.isna().sum().sum())
        # missing_values_location = group.isna()
        # missing_values_positions = missing_values_location.stack()[missing_values_location.stack()]
        # print(f"Locations of missing values:\n{missing_values_positions}")
        # return group

    # def categorize_blood_pressure(value):
    #     try:
    #         # Split the systolic and diastolic pressure
    #         systolic, diastolic = map(int, value.split('/'))
    #         #systolic, diastolic = map(object, value.split('/'))
    #         # Determine category based on conditions
    #         if systolic < 120 and diastolic < 80:
    #             return 0  # Normal
    #         elif 120 <= systolic < 130 and diastolic < 80:
    #             return 1  # Elevated
    #         elif (130 <= systolic < 140) or (80 <= diastolic < 90):
    #             return 2  # Stage 1 Hypertension
    #         elif systolic >= 140 or diastolic >= 90:
    #             return 3  # Stage 2 Hypertension
    #         else:
    #             return 5
    #     except ValueError:
    #         return None


if __name__ == "__main__":
    clinic_file = '/students/2024-2025/master/pre_eclampsy/pre-eclampsia.xlsx'
    group1, group2, group3, group4, group5, group6 = ClinicDataPreprocessing().load_data(clinic_file)
    # Example of finding missing value locations in group1
    ClinicDataPreprocessing().find_locations_missing_values_group1(group1)
    for group in [group2, group3, group4, group5, group6]:
        print(f'Group: {group}')
        ClinicDataPreprocessing().find_locations_missing_values_groups(group)
    
   
#     def imputer_missingvalue_of_group4_ML():
#         pipe1 = Pipeline([
#             ('scaler', StandardScaler()),  
#             ('pca', PCA()),                
#             ('RandomForestClassifier', RandomForestClassifier())  
#         ])

#         pipe2 = Pipeline([
#             ('scaler', StandardScaler()), 
#             ('pca', PCA()),                
#             ('Ridge', Ridge())  
#         ])
#         pipe1.fit(X_train, y_1_train)
#         pipe2.fit(X_train, y_2_train)
#         y_1_pred_train = pipe1.predict(X_train)
#         y_1_pred_test = pipe1.predict(X_test)
#         y_2_pred_train = pipe2.predict(X_train)
#         y_2_pred_test = pipe2.predict(X_test)
#         print("The Report at train set to evauate the model:")
#         print(classification_report(y_1_train, y_1_pred_train))
#         print("The predict values of the missing values:")
#         print( y_1_pred_test)
#         print("The Report at train set to evauate the Ridge model by R2:")
#         print(r2_score(y_2_train, y_2_pred_train))
#         print("The predict values of the missing values at 'bevalling na aantal dagen' cloumn :")
#         # Create an empty list to store the int values
#         int_predictions = []
#         for value in y_2_pred_test:
#             int_predictions.append(int(value)) 
#         print(int_predictions)
#         formatted_weeks_days = []
#         for value in int_predictions:
#             weeks = value // 7  
#             days = value % 7   
#             formatted_weeks_days.append(f'{weeks}+{days}') 
#         print(formatted_weeks_days)

#         y_1_missing = group4[group4['Sectio of vaginale geboorte?'].isna()].index
#         y_2_missing = group4[group4['bevalling na aantal dagen'].isna()].index
#         y_3_missing = group4[group4['bevalling na aantal weken + dagen'].isna()].index
#         for i, idx in enumerate(y_1_missing):
#             group4.at[idx, 'Sectio of vaginale geboorte?'] = y_1_pred_test[i]

#         for i, idx in enumerate(y_2_missing):
#             group4.at[idx, 'bevalling na aantal dagen'] = int_predictions[i]

#         for i, idx in enumerate(y_3_missing):
#             group4.at[idx, 'bevalling na aantal weken + dagen'] = formatted_weeks_days[i]
#         group4 = group4.astype({'bevalling na aantal dagen': int})
#         return group4

# if __name__ == "__main__":
#     clinic_file = '/students/2024-2025/master/pre_eclampsy/pre-eclampsia.xlsx'
#     group1, group2, group3, group4, group5, group6 = ClinicDataPreprocessing().load_data(clinic_file)
#     group_data = []
#     for group in [group1, group2, group3, group4, group5, group6]:
#         group_data.append(group['Patiëntcode'])
#         print(group_data)