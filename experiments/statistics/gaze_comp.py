import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Directories:
columns = ['file_number', 'frame_etg', 'frame_gar', 'X', 'Y', 'event_type', 'code']
main_dir = 'C:/Users/selloh/Desktop/Datasets/DREYEVE_DATA'
#output dir
output_dir = os.path.join('Transfer-Learning/Dreyeve', 'data')


#makes data directory if it doesnt exist
os.makedirs(output_dir, exist_ok=True)

def process_file(file_path):
    rows = []
    with open(file_path) as f:
        for line in f:
            line_data = line.strip().split()
            row = line_data  # Assuming the first element is foldername
            rows.append(row)
    return rows


def process_folder(foldername):
        folder_path = os.path.join(main_dir, foldername)
        print(f'processing folder: {foldername}')
        rows = []
        for filename in os.listdir(folder_path):
            if filename == 'etg_samples.txt':
                file_path = os.path.join(folder_path, filename)
                rows.extend(process_file(file_path))
        return foldername, rows

def main():
    dfs = []
    folders = [foldername for foldername in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, foldername))]
    
    with Pool() as pool:
        results = pool.map(process_folder, folders)
    
    for result in results:
        foldername, rows = result
        for row in rows:
            new_row = [foldername] + row
            dfs.append(new_row)
    
    df = pd.DataFrame(dfs, columns=columns)
    output_file = os.path.join(output_dir, 'gaze_data.csv')
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
