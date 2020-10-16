import numpy as np
import tempfile
from shutil import copy
import pandas as pd
import subprocess
import shutil
import timeit


def run_comparisons2(pair_id, metric_id):
    """This function runs through all required steps to compare a pair of maps (see Approach step 2-9)"""
    global map1_id, map2_id
    #map selection is different based on metric type
    if metric_id in single_comparison:
        map1 = map_list[pair_id][0]
        map2 = map_list[pair_id][1]        
    if metric_id in multi_comparison:
        map1 = map_pairs[pair_id][0]
        map2 = map_pairs[pair_id][1]
    # Stores map number of each map for later use in processing
    map1_id, map2_id = map_id(map1, map2)
    name1 = 'map' + str(map1_id)
    name2 = 'map' + str(map2_id)
    print(name1, name2, pair_id + 1)
    if metric_id == 'kappa':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files2(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/kappa.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\kappa.sts'
            frac, kappa = extract_stats(comparison, 0)
            return name1, name2, frac, kappa
    elif metric_id == 'fractal':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/fractal.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\\fractal.sts'
            sim1, sim2 = extract_stats(comparison, 4)
            if map1_id == map2_id:
                #if uneven number of maps assign dummy value
                sim2 = 999
            return name1, name2, sim1, sim2    
    else:
        return 'invalid metric id'
    
def copy_files2(path, map_nr1, map_nr2, metric_id):
    """This function takes a filepath and a pair of maps, turns them into asc format then copies the maps and other required files to the specified filepath"""
    #Convert maps
    map1name = 'map' + str(map_nr1) + '.asc'
    map2name = 'map' + str(map_nr2) + '.asc'
    map1 = "C:/LUMOS/MCK/ascmaps2/" + map1name
    map2 = "C:/LUMOS/MCK/ascmaps2/" + map2name
    copy(map1, path)   
    copy(map2, path)
    #Copy mask files
    copy(maskfile, path)   
    copy(masklandfile, path)
    #Generate and copy csl file for comparison
    if metric_id == 'kappa':
        csl_kappa(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    elif metric_id == 'fractal':
        csl_fractal(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    #Copy log file
    gen_log(path, map_nr1, map_nr2)
    #Copy legends folder
    src = 'C:/LUMOS/MCK/Legends'
    dst = path + '/Legends'
    shutil.copytree(src,dst)

def run_comparisons(pair_id, metric_id):
    """This function runs through all required steps to compare a pair of maps (see Approach step 2-9)"""
    global map1_id, map2_id
    #map selection is different based on metric type
    if metric_id in single_comparison:
        map1 = map_list[pair_id][0]
        map2 = map_list[pair_id][1]        
    if metric_id in multi_comparison:
        map1 = map_pairs[pair_id][0]
        map2 = map_pairs[pair_id][1]
    # Stores map number of each map for later use in processing
    map1_id, map2_id = map_id(map1, map2)
    name1 = 'map' + str(map1_id)
    name2 = 'map' + str(map2_id)
    print(name1, name2, pair_id + 1)
    if metric_id == 'kappa':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/kappa.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\kappa.sts'
            frac, kappa = extract_stats(comparison, 0)
            return name1, name2, frac, kappa
    elif metric_id == 'fractal':
        with tempfile.TemporaryDirectory() as tempdir:
            # Generate and copy all requires files to the temp folder
            copy_files(tempdir, map1_id, map2_id, metric_id)
            #call command prompt
            cslpath = tempdir + '/fractal.csl'
            logpath = tempdir + '/log.log'
            outputpath = tempdir
            call_cmd(mckdir=mckdir, csl=cslpath, log=logpath, outputdir=tempdir)
            #extract value from output file
            comparison = tempdir + '\\fractal.sts'
            sim1, sim2 = extract_stats(comparison, 4)
            if map1_id == map2_id:
                #if uneven number of maps assign dummy value
                sim2 = 999
            return name1, name2, sim1, sim2    
    else:
        return 'invalid metric id'
    
def extract_stats(sts, sts_id):
    """This function reads the output file generated by MCK and extracts the necessary stats based on the metric send"""
    if sts_id == 0:
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    #extract fraction correct
                    loc1 = line.find('correct="')
                    loc_start1 = loc1 + len('correct="')
                    loc_end1 =  line.find('"', loc_start1)
                    frac = float(line[loc_start1:loc_end1])
                    # Extract kappa
                    loc2 = line.find(' kappa="')
                    loc_start2 = loc2 + len(' kappa="')
                    loc_end2 =  line.find('"', loc_start2)
                    kappa = float(line[loc_start2:loc_end2])
                    return frac, kappa
    elif sts_id == 4:
        with open(sts, 'r') as file:
            for l, line in enumerate(file):
                if l == 3:
                    #Find first value
                    loc = line.find('overall_first="')
                    loc_start = loc + len('overall_first="')
                    loc_end =  line.find('"', loc_start)
                    sim1 = float(line[loc_start:loc_end])
                    #Find second value
                    loc2= line.find('overall_second="')
                    loc_start2= loc2 + len('overall_second="')
                    loc_end2= line.find('"', loc_start2)
                    sim2 = float(line[loc_start2:loc_end2])
                    return sim1, sim2
                        
    else:
        return 'Incorrect stats id'

def copy_files(path, map_nr1, map_nr2, metric_id):
    """This function takes a filepath and a pair of maps, turns them into asc format then copies the maps and other required files to the specified filepath"""
    #Convert csv maps to asc
    map1name = 'map' + str(map_nr1) + '.asc'
    map2name = 'map' + str(map_nr2) + '.asc'
    numpy2asc(map_data[map1_id], map1name, path, 10000.000000, 300000.000000, 100.000000, -32768 )   
    numpy2asc(map_data[map2_id], map2name, path, 10000.000000, 300000.000000, 100.000000, -32768 ) 
    #Copy mask files
    copy(maskfile, path)   
    copy(masklandfile, path)
    # Generate and copy csl file for comparison
    if metric_id == 'kappa':
        csl_kappa(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    elif metric_id == 'fractal':
        csl_fractal(path=path, map_nr1=map_nr1, map_nr2=map_nr2)
    # Copy log file
    gen_log(path, map_nr1, map_nr2)
    #Copy legends folder
    src = 'C:/LUMOS/MCK/Legends'
    dst = path + '/Legends'
    shutil.copytree(src,dst)
    
def map_id(map1, map2):
    """This function returns the map number for 2 provided maps based on the path of the maps"""
    map1_start = map1.find('_') + 1
    map1_end = map1.find('.')
    print(map1[map1_start:map1_end])
    map1_id = int(map1[map1_start:map1_end])
    
    map2_start = map2.find('_') + 1
    map2_end = map2.find('.')
    map2_id = int(map2[map2_start:map2_end]  )
    return map1_id, map2_id
    
def numpy2asc(ndarray, filename, savedir, xll, yll, cs, ndv, nan2num=False, delimiter=' '):
    """This function converts a numpy array into an asc file with header data"""
    #from https://stackoverflow.com/questions/24691755/how-to-format-in-numpy-savetxt-such-that-zeros-are-saved-only-as-0
    with open(savedir + '\\' + filename, 'w') as fh:
        
        #write header
        nrows, ncols = ndarray.shape
        header='\n'.join(["ncols " + str(ncols), 
                      "nrows " + str(nrows), 
                      'xllcorner ' + str(xll), 
                      'yllcorner ' + str(yll),
                      'cellsize ' + str(cs),
                      'NODATA_value ' + str(ndv),
                        ''])
        fh.write(header)
        
        #write each row in ndarray
        fmt="%.6e"
        fmti="%i"
        for row in ndarray:
            if nan2num:
                row[np.isnan(row)] = ndv
            line = delimiter.join(str(fmti % value) if value % 1 == 0 else str(fmt % value) for value in row)
            fh.write(line + '\n')
            
def csl_kappa(path, map_nr1, map_nr2):
    """This function copies a base csl file to the provided directory then rewrites it with desired changes"""
    file_dir = path + '\kappa.csl'
    copy(base_csl, path)
    with open(file_dir, 'w+') as file:
        file.truncate(0)
        file.write('<comparisonsets>\n')
        map1 = path + '\map' + str(map_nr1) + '.asc'
        map2 = path + '\map' + str(map_nr2) + '.asc'
        mask = path + '\mask.asc'
        file.writelines(['\n\t<comparisonset displayname="kappa"' + ' map1path=' + '"' \
                        + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Kappa"' \
                        + ' outputstatistics=' + '"'+ 'kappa.sts' + '"'+ ' theme1="maps" theme2="maps" up2date="0">' \
                        ,'\n\t\t<parameterset/>' \
                        , '\n\t\t<mask basemappath=' + '"' +  mask + '"' + ' displayname="NL_mask" mergeregions="0">' \
                        ,'\n\t\t\t<selectedregions>' \
                        ,'\n\t\t\t\t<value value="0"/>' \
                        ,'\n\t\t\t\t<value value="1"/>' \
                        ,'\n\t\t\t</selectedregions>' \
                        ,'\n\t\t</mask>' \
                        ,'\n\t</comparisonset>\n'])
        file.write('</comparisonsets>')
        
def csl_fractal(path, map_nr1, map_nr2):
    file_dir = path + '\\fractal.csl'
    copy(base_csl, file_dir)
    with open(file_dir, 'w+') as file:
        file.truncate(0)
        file.write('<comparisonsets>\n')
        map1 = path + '\map' + str(map_nr1) + '.asc'
        map2 = path + '\map' + str(map_nr2) + '.asc'
        mask = path + '\mask.asc'
        file.writelines(['\n\t<comparisonset displayname="fractal" map1path=' + '"' \
                        + map1 + '"'+ ' map2path=' + '"'+ map2 + '"'+ ' method="Moving Window based Structure"' \
                        + ' outputstatistics="fractal.sts" theme1="maps" theme2="maps" up2date="0">' \
                        ,'\n\t\t<parameterset>'\
                        , '\n\t\t\t<comparison_moving_window_structure aggregation="1" average_per_cell="1" ' \
                        + 'background_category="0" category_of_interest="0" display_map="0" ' \
                        + 'distance_weighed="0" halving="2" include_diagonal="1" metric="3" ' \
                        + 'per_category="0" radius="4" use_background="0"/>' \
                        ,'\n\t\t</parameterset>' \
                        , '\n\t\t<mask basemappath=' + '"' +  mask + '"' + ' displayname="NL_mask" mergeregions="0">' \
                        ,'\n\t\t\t\t<selectedregions>' \
                        ,'\n\t\t\t\t\t<value value="0"/>' \
                        ,'\n\t\t\t\t\t<value value="1"/>' \
                        ,'\n\t\t\t</selectedregions>' \
                        ,'\n\t\t</mask>' \
                        ,'\n\t</comparisonset>\n'])
        file.write('</comparisonsets>')    
        
def gen_log(path, map_nr1, map_nr2):
    """This function copies a base log file to the provided directory then rewrites it with desired changes"""
    copy(log_path, path)
    file_dir = path + '/log.log'
    line3 = 'maps=' + path + '\map' + str(map_nr1) + '.asc'
    line4 = 'maps=' + path + '\map' + str(map_nr2) + '.asc'
    #Read data
    with open(file_dir, 'r') as file:
        data = file.readlines()
    #Change Data    
    data[3] = line3 + '\n'
    data[4] = line4
    #Write new data in
    with open(file_dir, 'w') as file:
        file.writelines(data) 
        
def call_cmd(mckdir, csl, log, outputdir):
    """"""
    cmd= [mckdir, 'CMD',  '/RunComparisonSet', csl, log, outputdir]
    subprocess.run(cmd, check=True, shell=True)

#### SETUP VARIABLES ####    
# Folder that contain landusescanner output (csv format)
csvfolder = "C:/LUMOS/MCK/csvmaps/"
# Name of the csv files generated by landusescanner that contain map data
lus_outputname = "map500exp_"
ext = '.csv'
# Location of required files for MCK Comparisons
maskfile = 'C:/LUMOS/MCK/mask.asc'
masklandfile = 'C:/LUMOS/MCK/maskland.msk'
base_csl = 'C:/LUMOS/MCK/base_csl.csl'
log_path = 'C:/LUMOS/MCK/log.log'
mckdir = 'C:/Thesis_Python/Map_Comparison_Kit/MCK.exe'
# max map comparisons to run
nr_maps = 4


# Generate list containing a set of maps based on setup variables
maps = []
maps.extend([csvfolder + lus_outputname + str(x) + ext for x in np.arange(0, nr_maps)])

# Generate list containing all possible map comparisons
map_pairs=[]
for i in range(len(maps)):
    for j in range(len(maps)):
        if i < j:
            map1 = maps[i]
            map2 = maps[j]
            map_pairs.append((map1,map2))

# Generate list of map inputs for metrics that have a single map statistic
map_list= []
for i in np.arange(0, len(maps), 2):
        try:
            map1 = maps[i]
            map2 = maps[i + 1]
            map_list.append((map1,map2))
        except IndexError:
            # If there is only one map left send it in with itself
            map1 = maps[i]
            map2 = maps[i]
            map_list.append((map1,map2))  

# Store all map data once before making comparisons
map_data = []
map_data.extend([np.genfromtxt('C:/LUMOS/MCK/csvmaps/map500exp_'+str(x)+'.csv', delimiter=',') for x in np.arange(0,nr_maps)])

# Lists used in run_comparison to check comparison type
single_comparison = ['fractal', 'clump', 'nrpatch', 'simpson']
multi_comparison = ['kappa', 'kfuzzy', 'alloc', 'quant']

# for i in range(nr_maps):
#     dataname = 'C:/LUMOS/MCK/csvmaps/' + 'map500exp_' + str(i) + '.csv'
#     data = np.genfromtxt(dataname, delimiter=',')
#     filename = 'map' + str(i) +'.asc'
#     numpy2asc(data, filename, "C:/LUMOS/MCK/ascmaps2/", 10000.000000, 300000.000000, 100.000000, -32768 ) 
