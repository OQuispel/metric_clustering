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
    cm = conf_mat_old(m1, m2)
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

def conf_mat(map1, map2):
    """This function computes the confusion matrix of two maps."""
    #Maps are loaded in as nested arrays, undo this
    arr1 = map1.astype('int32')
    arr2 = map2.astype('int32')
    arr1 = np.concatenate(map1, axis=0)
    arr2 = np.concatenate(map2, axis=0)
    # compute confusion matrix
    cm = confusion_matrix(arr1, arr2)
    # delete the rows and columns for out of bound cells
    cm = np.delete(np.delete(cm, [24,27], 0), [24,27], 1)
    return cm