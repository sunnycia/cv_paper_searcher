# coding:utf-8
import urllib2
import requests
from bs4 import BeautifulSoup
import os

def main():
    root_link = 'http://openaccess.thecvf.com/'
    conference = 'ICCV'         # conference name
    year = 2017                 # conference year
    # save_path = './download'    # pdf save path
    save_path = 'document/'+conference.lower()+str(year)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


    from_page(root_link=root_link, conference=conference, year=year, save_path=save_path)

def from_page(root_link, conference, year, save_path):
    """
    Get all .pdf url from root_link
    """
    url = root_link + conference + str(year) + '.py'
    r = requests.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html5lib")
        index = 1
        print("\n=============== {0:10} ===============\n".format('Start Downloading'))
        for link in soup.find_all('a'):
            new_link = link.get('href')
            if new_link == None:
                continue
            if new_link.endswith('.pdf'):
                new_link = root_link + new_link
                print new_link
                download_file(new_link, save_path)
                index += 1
        print('Totally {} files have been downloaded.'.format(index))
    else:
        print("ERRORS occur !!!")

def download_file(download_url, save_path):
    """
    Download pdf file from download_url
    """
    try:
        response = urllib2.urlopen(download_url)
    except urllib2.HTTPError:
        print("url is not exist")
    else:
        file_name = download_url.split('/')[-1]
        save_name = os.path.join(save_path, file_name)
        file = open(save_name, 'w')
        file.write(response.read())
        file.close()
        print("Completed Dowloaded: {0:30}".format(file_name))

if __name__ == "__main__":
    main()