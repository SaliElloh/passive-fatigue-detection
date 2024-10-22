import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Directories:

main_dir = '/home/selloh/externalProjects/TransferLearning/datasets/DREYEVE_DATA'
#output dir
output_dir = os.path.join('/home/selloh/externalProjects/TransferLearning/Dreyeve', 'data')
#makes data directory if it doesnt exist
os.makedirs(output_dir, exist_ok=True)


def process_file(file_path):
    rows = []
    with open(file_path) as f:
        for line in f:
            line_data = line.strip().split()
            row = line_data  
            rows.append(row)
    return rows


def process_folder(foldername):
        folder_path = os.path.join(main_dir, foldername)
        print(f'processing folder: {foldername}')
        rows = []
        for filename in os.listdir(folder_path):
            if filename == 'speed_course_coord.txt':
                file_path = os.path.join(folder_path, filename)
                rows.extend(process_file(file_path))
        return foldername, rows

def main():
    dfs = []
    columns = ['file_number', 'frame_num', 'speed', 'course', 'lat', 'lon']
    folders = [foldername for foldername in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, foldername))]
    
    with Pool() as pool:
        results = pool.map(process_folder, folders)
    
    for foldername, rows in results:
        for row in rows:
            if len(row) == 3:
                new_row = [foldername] + row + [0 , 0]
            else:
                new_row = [foldername] + row
                print(new_row)
                print(f'appending data for folder {foldername} ')
            dfs.append(new_row)
    
    df = pd.DataFrame(dfs, columns=columns)
    output_file = os.path.join(output_dir, 'speed_data.csv')
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
