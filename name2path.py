#coding=utf-8
import os
import glob
import cPickle as pkl

paper_folder_list = ['document/cvpr2017', 'document/cvpr2018', 'document/eccv2018']

name_dict = {}

for paper_folder in paper_folder_list:
    paper_path_list = glob.glob(os.path.join(paper_folder, '*.*'))
    for paper_path in paper_path_list:
        paper_name = os.path.splitext(os.path.basename(paper_path))[0]
        if paper_name not in name_dict:
            # 这里只加了路径信息，后面可能会按需求增加内容
            name_dict[paper_name] = paper_path
        else:
            print paper_name, "already exists."

pkl.dump(name_dict, open('database/name_dict.pkl', 'wb'))
# for key in name_dict:
#     print key, name_dict[key]