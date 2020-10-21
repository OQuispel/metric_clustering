import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import pontius as pont
import multiprocessing.pool as mpp
import scipy.spatial.distance as ssd

def gen_cms(pair_id):
    """Test function to generate some CMs to analyze"""
    map1 = pontius_pairs[pair_id][0]
    map2 = pontius_pairs[pair_id][1]
    m1_dir = abs_dir + 'ascmaps/' + map1 + '.asc'
    m2_dir = abs_dir + 'ascmaps/' + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    
    rows_cols = ['class' + str(i + 1) for i in range(28)]
    rows_cols.remove('class24')
    rows_cols.remove('class28')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    df_pont = pont.pontiPy(df)
    return df_pont

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def calc_multi(pair_id, class1=22):
    """This function is the basis for calculating Pontius' metrics (2014), starting off by computing the confusion matrix for two provided maps and 
    ending with the Total Disagreement if it is assigned to be calculated."""
    map1 = pontius_pairs[pair_id][0]
    map2 = pontius_pairs[pair_id][1]
    m1_dir = abs_dir + 'ascmaps/' + map1 + '.asc'
    m2_dir = abs_dir + 'ascmaps/' + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    # turn array into df so that ppy can work with it
    rows_cols = ['class' + str(i) for i in range(28)]
    rows_cols.remove('class23')
    rows_cols.remove('class27')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    #create object to run ppy code
    df_pont = pont.pontiPy(df)
    #calculate overall accuracy
    oa = calc_oa(df_pont)
    #calculate pontius metrics
    od, oqd, oad = calc_pontius(df_pont)
    #calculate categorical overall quantity difference
    oqd_n = calc_oqd_n(df_pont, class1)
    return map1, map2, oa, od, oqd, oad, oqd_n

def calc_classes(pair_id, class1=22):
    """This function goes through all steps to turn a pair of maps into metric values for PLAND, TCA, Allocation Disagreement and Exchange Disagreement"""
    map1 = pontius_list[pair_id][0]
    map2 = pontius_list[pair_id][1]
    m1_dir = abs_dir + 'ascmaps/' + map1 + '.asc'
    m2_dir = abs_dir + 'ascmaps/' + map2 + '.asc'
    m1 = np.loadtxt(m1_dir, skiprows=6)
    m2 = np.loadtxt(m2_dir, skiprows=6)
    #compute confusion matrix
    cm = conf_mat(m1, m2)
    # turn array into df so that ppy can work with it
    rows_cols = ['class' + str(i) for i in range(28)]
    rows_cols.remove('class23')
    rows_cols.remove('class27')
    df = pd.DataFrame(data=cm, index=rows_cols, columns=rows_cols)
    #create object to run ppy code
    df_pont = pont.pontiPy(df)
    #calc PLAND
    pland1 = df_pont.size(class1, 'X') / df_pont.size() * 100
    pland2 = df_pont.size(class1, 'Y') / df_pont.size() * 100
    #calc TCA
    tca1 = df_pont.size(class1, 'X')
    tca2 = df_pont.size(class1, 'Y')
    return pland1, pland2, tca1, tca2


def conf_mat(map1, map2):
    """This function computes the confusion matrix of two maps."""
    # change array to lists as confusion matrix library doesn't work on arrays
    list1 = np.ndarray.tolist(map1)
    list2 = np.ndarray.tolist(map2)
    # turn list of lists into single list and compute confusion matrix
    l1 = flatten(list1)
    l2 = flatten(list2)
    cm = confusion_matrix(l1, l2)
    # delete the rows and columns for out of bound cells
    cm = np.delete(np.delete(cm, [24,27], 0), [24,27], 1)
    return cm

def calc_oa(cm):
    """This function calculates Overall Accuracy from the confusion Matrix
    OA = diagonal of matrix/total nr cells"""  
    #Calculate OA
    oa = cm.agreement() / cm.size()
    return oa

def calc_pontius(cm):
    """This function calculates the overall values of the Pontius metrics and returns these. Note that total allocation is returned and not 
    the newer components of Shift and Exchange."""
    #Calculate overall difference
    od = cm.difference()
    #Calculate overall quantity difference
    oqd = cm.quantity()
    #Calculate total allocation difference as the components of shift and exchange together
    oad = cm.shift() + cm.exchange()
    return od, oqd, oad

def calc_oqd_n(cm, n):
    """This function calculates the overall quantity difference for a provided class n"""
    oqd_n = cm.quantity(n)
    return oqd_n

def single_df(maps, stats, metric):
    """This function takes the output from the run comparisons function for single map comparison metrics and returns the distance matrix"""
    #create base df to form basis for distance matrix
    df = pd.DataFrame(index=maps)
    df[metric] = stats
    #calculate euclidean distance between all values then change to matrix form
    matrix = ssd.squareform(ssd.pdist(df))
    df_clean = pd.DataFrame(matrix, index=maps, columns=maps)
    
    # save values to disk
    csv_val = csv_dir + metric + '_values.csv'
    df_vals = pd.DataFrame(index=maps)
    df_vals[metric] = stats
    df_vals.to_csv(csv_val)
    #save df to disk
    csv_name = csv_dir + metric + '_df.csv'
    df_clean.to_csv(csv_name, index=False)
    
def multi_df(map1, map2, stats, metric):
    """This function takes the output from the run calc_poontius function for multi map comparison metrics and stores the data to disk"""
    #Create two dataframes with swapped map columns
    df = pd.DataFrame()
    df['map1'] = [x for x in map1]
    df['map2'] = [x for x in map2]
    df[metric] = stats
    df2 = df
    df2 = df2[[ 'map2', 'map1', metric]]
    df2.columns = ['map1', 'map2', metric]      
    df_concat = pd.concat([df, df2])
    df_pivot = df_concat.pivot(index='map2', columns='map1', values=metric)   
    ## clean up the presentation
    #Remove unecessary labeling
    index = df_pivot.index.union(df_pivot.columns)
    df_clean = df_pivot.reindex(index=index, columns=index)
    #reindex to correct numerical order
    ordered = df_clean.index.to_series().str.rsplit('p').str[-1].astype(int).sort_values()
    df_clean = df_clean.reindex(index=ordered.index, columns=ordered.index).fillna(1).round(decimals=3) 
    #save df to disk
    csv_name = csv_dir + metric + '_df.csv'
    df_clean.to_csv(csv_name, index=False)

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap
    
### VARIABLES ###
# max map comparisons to run
nr_maps = 5

## ABSOLUTE PATH ##
# changing this variable to where the main folder with all files is is the only path adjustment that is necessary
#abs_dir = 'C:/LUMOS/clusterfiles/'
#use for 100 map test
abs_dir = 'C:/LUMOS/MCK/'
#directory where the dataframes containing output are stored
csv_dir = abs_dir + 'Output_DFs/'
# Generate list containing all possible map comparisons
pontius_pairs=[]
for i in range(nr_maps):
    for j in range(nr_maps):
        if i < j:
            map1 = 'map' + str(i)
            map2 = 'map' + str(j)
            pontius_pairs.append((map1,map2))

maps = ['map' + str(i) for i in range(nr_maps)]
# Generate list of map inputs for metrics that have a single map statistic
pontius_list= []
for i in np.arange(0, nr_maps, 2):
        try:
            map1 = maps[i]
            map2 = maps[i + 1]
            pontius_list.append((map1,map2))
        except IndexError:
            # If there is only one map left send it in with itself
            map1 = maps[i]
            map2 = maps[i]
            pontius_list.append((map1,map2)) 
            
#Variable that stores number of iterations required for the assigned number of maps  
multi_its = len(pontius_pairs)
single_its = len(pontius_list)