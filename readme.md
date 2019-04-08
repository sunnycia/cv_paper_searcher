# What this repo about?

this repo is a simple cv paper(CVPR13~18; ICCV13,15,18; ECCV18) finder. You can easily find out the paper with the most key word you searched.

This repo also provide the link of  above CV paper and sentence/word dictionary in python pickle form.


# Get start

A few steps have to be done before you start to use this tool.

## Prepare database

If you want to use the above mentioned cv papers as the database, you can download from this [link]() and put those .pkl file in the *database* folder and the papers in the *document* folder.

You may interested in how to scrape the cv papers, [this link]() provide several ways.

If you want to use your own paper dataset, put them into a folder, say your_folder. Run updata_database.py to build the paper database.
```python
python update_database.py --document_folder=your_folder
```


## Search the papers with key word
Currently this function is far from complete and intelligent, because it only support one word searching, but at least it works. Try to use a key word of *saliency* if you want to get some paper about the saliency
```python
python index.py --keyword=saliency --rank=50
```
A report with name of report_saliency.md will be generated, containing the paper names and the sentences with the keyword. 

## TODO

