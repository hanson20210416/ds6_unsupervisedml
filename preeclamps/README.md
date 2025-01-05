Author = Zhipeng He 

All explanations are on the Jupiter notebook file.

The clinicdata_preprocessing.py preprocesses the clinic data. Likes to extract the patientcode.

The proteindata_preprocessing.py is to preprocess the protein data. transpose the dataframe, rename the rows, drop some columns, and impute the missing values.

The data_filter.py introduces three methods to filter out some proteins by setting a threshold.
    method1: Hemoglobin proteins.
    method2: The most stable expression proteins.
    method3: Just keep Some proteins that have big different expressions between different groups to find the biomarkers.
    
The class_dataprocessByCluster.py imports some cluster algorithms from sklearn package to cluster the filtered_data and evaluate the results.

The Visualization.py makes some functions to display PCA,tSNE,UMAP plot, and for the filtered_data, do scaler and PCA.

The class_dataprocessByScanpy.py is another method to cluster the data, but the cluster result is not good. So I just imported it and showed the results at the end of the notebook.
