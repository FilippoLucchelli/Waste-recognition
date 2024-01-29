import os

def list_and_sort_paths(folder):
    names_list=sorted(os.listdir(folder))
    paths_list=[os.path.join(folder, names_list[i]) for i in range(len(names_list))]
    return paths_list