#coding=utf-8
import os
import cPickle as pkl
from load_dict import *
from pdf2words import text2sents,pdf2text
import operator
import string
import argparse
import shutil
# from shutil import copyfile

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, required=True, help='Key word you want to search')
    parser.add_argument('--sent_length', type=int, default=200, help='instances threshold.')
    parser.add_argument('--threshold', type=int, default=3, help='instances threshold.')
    parser.add_argument('--rank', type=int, default=10, help='Get top rank paper.')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

threshold=args.threshold
sent_length = args.sent_length
rank = args.rank
reverse_dict_path = 'database/reverse_dict.pkl'
dict_path = 'database/dict.pkl'
sentence_dict_path = 'database/sentence_dict.pkl'
name_dict_path = 'database/name_dict.pkl'
reverse_dict = load_dict(reverse_dict_path)
sentence_dict = load_dict(sentence_dict_path)
name_dict = load_dict(name_dict_path)
# print name_dict;exit()
key_word = args.keyword
key_word = key_word.lower()

phrase_flag=False
multiword_flag=False
if ' ' in keyword:
    phrase_flag=True
if ',' in keyword:
    multiword_flag=True

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
            sent = sent.lower()
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
        sent_count = 0
        for sent in sent_list:
            if ' %s ' % key_word in sent.lower():
                sent_count += 1
                sent = sent.replace("\r\n", " ")               
                sent = sent.replace("\r", " ")               
                sent = sent.replace("\n", " ")               
                sent = sent.replace("- ", "")               
                sent = sent.replace("|", " ")

                # 限制句子输出长度 
                if len(sent) < sent_length+10:
                    pass
                else:
                    index = sent.find(key_word)
                    start = index - int(sent_length/2)
                    if start < 0:
                        start = 0
                    end = start + sent_length
                    sent = sent[start:end]
                # # 限制句子中的奇怪字符
                # "".join(filter(lambda char: char in string.printable, sent))
                if flag:
                    report.write('|{}|{}|...{}...|\n'.format('', '', sent))
                else:
                    report.write('|{}|{}|...{}...|\n'.format(str(paper_count), paper_name, sent))
                flag = True

        report.write('||{} instances in total. (in {})|\n'.format(str(sent_count), os.path.basename(os.path.dirname(paper_path))))
        paper_count += 1

    ## get paper
    output_dir = '{}_rank_{}'.format(key_word, str(rank))
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for (paper_name, cnt) in sent_count_list[:rank]:
        paper_path = name_dict[paper_name]
        shutil.copyfile(paper_path, os.path.join(output_dir, str(cnt)+'_'+os.path.basename(paper_path)))




report.close()