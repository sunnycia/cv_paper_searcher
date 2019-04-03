import os
import cPickle as pkl
from load_dict import *
from pdf2words import text2sents,pdf2text

reverse_dict_path = 'database/reverse_dict.pkl'
dict_path = 'database/dict.pkl'
name_dict_path = 'database/name_dict.pkl'
reverse_dict = load_dict(reverse_dict_path)
name_dict = load_dict(name_dict_path)


key_word = 'fusion'
report_path = 'report_{}.md'.format(key_word)
report = open(report_path, 'wb')
report.write('|Index|Title|sentenct|\n')
report.write('|---|---|---|\n')

if not key_word in reverse_dict:
    print key_word, "not exist."
else:
    paper_list = reverse_dict[key_word]
    print paper_list, len(paper_list)

    # generate report
    count = 0
    for paper_name in paper_list:
        print paper_name
        try:
            paper_path = name_dict[paper_name]
        except:
            continue
        text = pdf2text(paper_path)
        sent_list = text2sents(text)
        flag = False
        for sent in sent_list:
            if key_word in sent:
                sent = sent.replace("\n", "")
                # str2 = str.replace("\n", "")
                # print "info:", sent
                if flag:
                    report.write('|{}|{}|{}|\n'.format('', '', sent))
                else:
                    report.write('|{}|{}|{}|\n'.format(str(count), paper_name, sent))
                flag = True
        count +=1
        # if count == 10:
        #     report.close();exit()

report.close()