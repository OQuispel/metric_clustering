# metric_clustering
Contains scripts used in the process of clustering land use maps based on similarity metrics


cmdprompt_mck.ipynb runs all desired metrics from MCK via command prompt and processes the results

convertasc.ipynb is used to convert csvs to asc format which is needed for use in MCK

generate_csl.ipynb generates the CSLs for the required metrics, which is a file describing what comparisons MCK should make

generate_mask.ipynb is used to generate the mask file needed to filter out undesired areas of the map, this is done by converting a random output map from the land use scanner model to the desired format
