#coding=utf-8
import os
import cPickle as pkl
from load_dict import *
from pdf2words import text2sents,pdf2text
import operator

import argparse



threshold=7
reverse_dict_path = 'database/reverse_dict.pkl'
dict_path = 'database/dict.pkl'
sentence_dict_path = 'database/sentence_dict.pkl'
name_dict_path = 'database/name_dict.pkl'
reverse_dict = load_dict(reverse_dict_path)
sentence_dict = load_dict(sentence_dict_path)
name_dict = load_dict(name_dict_path)
# print name_dict;exit()
key_word = 'fusion'
report_path = 'report_{}.md'.format(key_word)
report = open(report_path, 'wb')
report.write('|Index|Title|sentence|\n')
report.write('|---|---|---|\n')

# def tuple_sort(tuple_list):


if not key_word in reverse_dict:
    print key_word, "not exist."
else:
    paper_list = reverse_dict[key_word]
    print paper_list, len(paper_list)

    # generate report
    sent_count_list = []
    for paper_name in paper_list:
        print paper_name
        try:
            paper_path = name_dict[paper_name]
        except:
            continue

        ## 在线读取太慢，改为从database中读取
        # text = pdf2text(paper_path)
        # sent_list = text2sents(text) 
        if paper_name not in sentence_dict:
            print paper_name, "not in sentence_dict, please check."
            exit()
        sent_list = sentence_dict[paper_name]
        filter_sent_list = []
        for sent in sent_list:
            if ' %s ' % key_word in sent:
                # sent = sent.replace("\n", " ")
                # str2 = str.replace("\n", "")
                # print "info:", sent
                filter_sent_list.append(sent)
        if len(filter_sent_list) < threshold:
            continue
        sent_count_list.append((paper_name, len(filter_sent_list)))

    sent_count_list.sort(key=operator.itemgetter(1))
    sent_count_list.reverse()
    ## SORT
    paper_count = 1
    for (paper_name, _) in sent_count_list:
        paper_path = name_dict[paper_name]
        sent_list = sentence_dict[paper_name]
        flag = False
        sent_count = 1
        for sent in sent_list:
            if ' %s ' % key_word in sent:
                sent = sent.replace("\n", " ")               
                sent = sent.replace("- ", "")               
                if flag:
                    report.write('|{}|{}|{}|\n'.format('', '', sent))
                else:
                    report.write('|{}|{}|{}|\n'.format(str(paper_count), paper_name, sent))
                flag = True
                sent_count += 1

        report.write('||{} instances in total. (in {})|\n'.format(str(sent_count), os.path.basename(os.path.dirname(paper_path))))
        # report.write('||{} instances in total. (in {})|\n'.format(str(sent_count), 'haha'))
        # print paper_path, os.path.basename(os.path.dirname(paper_path))
        paper_count += 1
        # if count == 10:
        #     report.close();exit()

report.close()