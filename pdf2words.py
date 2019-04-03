#coding=utf-8

from cStringIO import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import os
import re
import sys, getopt
import nltk
from nltk.book import *
from nltk.corpus import stopwords
import unicodedata
#converts pdf, returns its text content as a string
def pdf2text(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)

    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = file(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    return text 

# def hasNumbers(inputString):
#     return any(char.isdigit() for char in inputString)
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))
def hasSpecialChar(inputString):
    return not inputString.isalpha()

def text2sents(text):
    # print text;exit()
    sent_list = nltk.sent_tokenize(text.decode('utf-8'))
    for i in range(len(sent_list)):
        sent_list[i] = unicodedata.normalize('NFKD', sent_list[i]).encode('ascii', 'ignore')

    return sent_list

def text2words(text):
    word_list = nltk.word_tokenize(text.decode('utf-8'))
    # 转换为小写, Unicode to ascii string,  去掉含特殊符号的词
    for i in range(len(word_list)-1,0,-1):
        word_list[i] = word_list[i].lower() 
        word_list[i] = unicodedata.normalize('NFKD', word_list[i]).encode('ascii','ignore')
        if hasSpecialChar(word_list[i]):
            del word_list[i]

    # # 去掉含数字的词
    # for i in range(len(word_list)-1, 0, -1):
    #     if hasNumbers(word_list[i]):
    #         del word_list[i]
    
    # 去掉重复元素
    word_list = list(set(word_list))

    # 去掉过短的词  去掉停止词
    stop_words = set(stopwords.words('english'))
    for i in range(len(word_list)-1, 0, -1):
        if len(word_list[i]) <3:
            del word_list[i]
            continue
        if word_list[i] in stop_words:
            del word_list[i]

    # lemmatization, 词形还原
    ##pass

    return word_list
if __name__ == '__main__':
    # 读pdf文件，转换为text
    text = pdf2text('atest.pdf')
    # 从text生成关键词list
    filtered_words = text2words(text)
    
    for word in filtered_words:
        print word
    print len(words), len(filtered_words)



    # 词频统计
    # FreqDist()获取在文本中每个出现的标识符的频率分布
    # fq_dist = FreqDist(words)
    # print(fq_dist)
    # # 词数量
    # print(fq_dist.N())
    # # 不重复词的数量
    # print(fq_dist.B())


# import PyPDF2
# pdf_file = open('eccv2018/Alexander_Vakhitov_Stereo_relative_pose_ECCV_2018_paper.pdf', 'rb')
# read_pdf = PyPDF2.PdfFileReader(pdf_file)
# number_of_pages = read_pdf.getNumPages()
# print '%s pages in total.' % str(number_of_pages)
# page = read_pdf.getPage(8)
# page_content = page.extractText()
# print page_content
# # print page_content.encode('utf-8')


# from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
# from pdfminer.converter import TextConverter
# from pdfminer.layout import LAParams
# from pdfminer.pdfpage import PDFPage
# from io import StringIO

# def convert_pdf_to_txt(path):
#     rsrcmgr = PDFResourceManager()
#     retstr = StringIO()
#     codec = 'utf-8'
#     laparams = LAParams()
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
#     fp = open(path, 'rb')
#     interpreter = PDFPageInterpreter(rsrcmgr, device)
#     password = ""
#     maxpages = 0
#     caching = True
#     pagenos=set()

#     for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
#         interpreter.process_page(page)

#     text = retstr.getvalue()

#     fp.close()
#     device.close()
#     retstr.close()
#     return text


# text = convert_pdf_to_txt('atest.pdf')
# print text