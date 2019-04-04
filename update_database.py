#coding=utf-8
import os, glob
from pdf2words import *
from load_dict import * 
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_folder', type=str, required=True, help='document folder')
    return parser.parse_args()

print "Parsing arguments..."
args = get_arguments()

def reverse_dictionary(dictionary):
    reverse_dict = {}
    
    for key in dictionary:
        key_word_list = dictionary[key]
        for key_word in key_word_list:
            # print key_word
            if key_word in reverse_dict:
                if key in reverse_dict[key_word]:
                    continue
                else:
                    reverse_dict[key_word].append(key)
            else:
                reverse_dict[key_word] = []
                reverse_dict[key_word].append(key)

    return reverse_dict

dict_path = 'database/dict.pkl'
reverse_dict_path = 'database/reverse_dict.pkl'

sentence_dict_path = 'database/sentence_dict.pkl'
def update_word_sent_database(folder):
   # 加载旧字典
    word_dictionary = load_dict(dict_path)
    sent_dictionary = load_dict(sentence_dict_path)

    # 更新字典
    for pdf_path in glob.glob(os.path.join(folder, '*.*')):
        print "Processing", pdf_path
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        if pdf_name in sent_dictionary:
            print pdf_name, 'already in sent_dictionary!'
            continue
        else:
            text = pdf2text(pdf_path)
            if text == -1:
                print "Warning:", pdf_name, "is corruptted."
                continue
            # 从text生成句子list
            sentence_list = text2sents(text)
            sent_dictionary[pdf_name] = sentence_list

            if pdf_name in word_dictionary:
                print pdf_name, 'already in word_dictionary!'
                continue
            else:
                filtered_words = text2words(text)
                word_dictionary[pdf_name] = filtered_words
        print "Done for", pdf_path, ',%d words in total.' % len(filtered_words)
        print "Done for", pdf_path, ',%d sentences in total.' % len(sentence_list)

    # 生成word_dictionary倒排
    reverse_dict = reverse_dictionary(word_dictionary)

    # 保存 字典以及倒排字典
    pkl.dump(word_dictionary, open(dict_path, 'wb'))
    pkl.dump(reverse_dict, open(reverse_dict_path, 'wb'))

    pkl.dump(sent_dictionary, open(sentence_dict_path, 'wb'))


'''
def update_word_database(folder):
    # 加载旧字典
    dictionary = load_dict(dict_path)

    #更新字典
    for pdf_path in glob.glob(os.path.join(folder, '*.*')):
        print "Processing", pdf_path
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        if pdf_name in dictionary:
            print pdf_name, 'already in dictionary!'
            continue

        text = pdf2text(pdf_path)
        # 从text生成关键词list
        filtered_words = text2words(text)

        # 建字典
        dictionary[pdf_name] = filtered_words
        print "Done for", pdf_path, ',%d words in total.' % len(filtered_words)

    # 生成倒排
    reverse_dict = reverse_dictionary(dictionary)

    # 保存 字典以及倒排字典
    pkl.dump(dictionary, open(dict_path, 'wb'))
    pkl.dump(reverse_dict, open(reverse_dict_path, 'wb'))
def update_sent_database(folder):
    # 加载旧字典
    dictionary = load_dict(sentence_dict_path)

    # 更新字典
    for pdf_path in glob.glob(os.path.join(folder, '*.*')):
        print "Processing", pdf_path
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if pdf_name in dictionary:
            print pdf_name, 'already in dictionary!'
            continue

        text = pdf2text(pdf_path)
        sentence_list = text2sents(text)

        dictionary[pdf_name] = sentence_list
        print "Done for", pdf_path, '%d sentences in total.' % len(sentence_list)

    pkl.dump(dictionary, open(sentence_dict_path, 'wb'))
'''


if __name__=='__main__':
    pdf_folder = args.document_folder

    # update_word_database(pdf_folder)
    update_word_sent_database(pdf_folder)