import os


def create_folders(file_path,total_folders):
    for i in range(total_folders):
        os.mkdir(file_path + 'sample_' + str(i))