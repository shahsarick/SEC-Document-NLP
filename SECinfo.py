# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 08:06:25 2016

@author: Sarick SHAHS
"""
import requests
import os
os.chdir("C:\Users\Sarick\Documents\Python Scripts")
#%%
import urllib2
url = "http://www.secinfo.com/d14D5a.wMUmw.htm"
response = urllib2.urlopen(url)
main_doc = response.read() #read the SEC info
#%%

#updated the code to now use comments on a new line
from bs4 import BeautifulSoup #import stuff
soup = BeautifulSoup(main_doc,'html.parser')
soup = soup.get_text()
soup2 = soup.encode('utf-8').split('\n')

'''
Next step is to see if i can download all of them myself and save as text files!
Then after that start NLP'ing that bitch
'''



#%%
url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=318154&type=10-K%25&dateb=&owner=exclude&start=0&count=100"
r = requests.get(url)
soup = BeautifulSoup(r.text) # Initializing to crawl again
table = soup.find('table', 'tableFile2')
linkList = table.find_all('a', id='documentsbutton')
#%%

#specifically looking for 10k's
cik = "318154"
filing_type = "10-K"
start = "0"
base_url = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={0}&type={1}%25&dateb=&owner=exclude&start={2}&count=100".format(cik, filing_type, start)
r = requests.get(base_url)
soup = BeautifulSoup(r.text) # Initializing to crawl again
table = soup.find('table', 'tableFile2')
linkList = table.find_all('a', id='documentsbutton')

for link in linkList:
    print link.get('href')
    #self.parse_filing_index(ticker, filing_type, l)
#%%
import os
os.chdir('C:\Users\Sarick\Desktop')
import sas7bdat
from sas7bdat import SAS7BDAT
with SAS7BDAT('filings.sas7bdat') as f:
    df = f.to_data_frame()

#%%
df.to_csv('df.csv')
df = pd.read_csv('df.csv')
#%%
df2 = df[df['formtype']=='DEFM14A']

os.chdir('C:\Users\Sarick\Documents\Python Scripts\SEC_Filings_Download')
for i in df2['filename']:
    response = urllib2.urlopen("https://www.sec.gov/Archives/"+i)
    with open(i[-24:], 'w') as f:
        f.write(response.read())
#%%
import urllib2
import requests

url = 'https://www.sec.gov/Archives/edgar/data/1056258/0000728672-99-000044.txt'
#response = urllib2.urlopen(url)
x = requests.get(url)
#webContent = response.read()
#webContent
re.sub('\s+', ' ', x.text)
re.sub('\n', '', x.text)
#%%
import urllib2
url = 'https://www.sec.gov/Archives/edgar/data/1056258/0000728672-99-000044.txt'
response = urllib2.urlopen(url)
with open('output.txt', 'w') as f:
    f.write(response.read())
#%%
import glob
import pickle
files = glob.glob('C:\\Users\\Sarick\\Documents\\Python Scripts\\SEC_Filings_Download\\*')
list1 = []
for i in files:
    with open(i) as f:
        file1 = f.read()
        list1.append(file1)
#%%
import pickle
os.chdir("C:\Users\Sarick\Documents\Python Scripts\SEC_Filings_Download")
pickle.dump(list1, open( "list1.pkl", "wb" ))
list1 = pickle.load(open("list1.pkl", "rb"))
#%%
#Remove \n
import re
list2 = []
for i in list1:
    list2.append(re.sub('\n', '', i))

import pandas as pd
df= pd.DataFrame(list2)
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english", 
                        token_pattern="\\b[a-zA-Z][a-zA-Z]+\\b", 
                        min_df=10)
vecs = tfidf.fit_transform(list2)
#%%
from sklearn.decomposition import NMF
nmf = NMF(n_components=300)
nmf_vecs = nmf.fit_transform(vecs)
#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=30,  n_jobs=-1)
cluster_labels = kmeans.fit_predict(nmf_vecs)
#%%
from sklearn import metrics
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(nmf_vecs, kmeans.labels_, sample_size=1000))
#%%
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
range_n_clusters = [5, 10, 15, 20, 25, 30]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(nmf_vecs) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(nmf_vecs)
    silhouette_avg = silhouette_score(nmf_vecs, cluster_labels)
    print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(nmf_vecs, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples
        
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(nmf_vecs[:, 0], nmf_vecs[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)
    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    
#%% 
import gensim
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model_pred = model[list2]
