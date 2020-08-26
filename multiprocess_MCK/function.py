import subprocess

def run_slice(csl, log):
    cmd= ['C:/Thesis_Python/Map_Comparison_Kit/MCK.exe', 'CMD',  '/RunComparisonSet', csl, log, 'C:\LUMOS\MCK\Output']
    subprocess.run(cmd, check=True, shell=True)