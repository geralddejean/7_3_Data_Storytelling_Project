import requests
import zipfile
import io
import ssl
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from os import path
import matplotlib.pyplot as plt
import seaborn as sns

#open .arff file if it exists in directory or fetch it from the web
arff_file = '2year.arff'

if path.exists(arff_file) == False:
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip'
    r = requests.get(url)
    textfile = zipfile.ZipFile(io.BytesIO(r.content))
    textfile.extract(arff_file)

file = open(arff_file, 'r')
filedata = file.read()
filedata = filedata.replace('class {0,1}','Attr65 numeric')

file = open(arff_file, 'w')
file.write(filedata)
file.close()

#Convert .arff file to a dataframe
data = loadarff(arff_file)
df = pd.DataFrame(data[0])

# Show relavant statistics
allStats = df.describe(include='all')

# Show relevant statistics with outliers removed
df_NO = df.loc[:,'Attr1':'Attr64']
df_NO = df_NO[(df_NO >= df_NO.mean()-2*df_NO.std()) &
                        (df_NO <= df_NO.mean()+2*df_NO.std())]
df_NO['Target'] = df['Attr65']
allStats_NO = df_NO.describe(include='all')

#Fill all missing values with the mean (df1)
df1 = df_NO.fillna(df_NO.mean())

#Remove rows with any Nan values (df2)
df2 = df_NO.dropna().reset_index(drop=True)

# Choose new dataframe to be df1 or df2
df_NO = df1

#Show correlation matrix to see if Attr37 is highly correlated with anything
corrMat = df_NO.corr()
corrMat['Target'].nsmallest(1), corrMat['Target'].nlargest(2)

# Create a dataframe of attributes and their meanings
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data'
url_get = requests.get(url)
soup = BeautifulSoup(url_get.content, 'lxml')
with open('attributes.txt', 'w', encoding='utf-8') as f_out:
    f_out.write(soup.prettify())
    f_out.close()

with open("attributes.txt", "r") as f:
    lines = f.readlines()
    lines = lines[367:494]
    f.close()
    
finance_def_df = pd.DataFrame({'expression':lines})
finance_def_df = finance_def_df[finance_def_df.index%2 == 0]
finance_def_df = finance_def_df.replace('^\s{6}X[0-9]+\t','',regex=True)\
.replace('$\n','',regex=True)
finance_def_df.index = df.loc[:,'Attr1':'Attr64'].columns

# Create DataFrame of strong correlations (negative and positive) based on correlation threshold
strongCorrMatrix = corrMat.unstack().reset_index()
strongCorrMatrix.rename(columns={'level_0':'Pair1',
                                 'level_1':'Pair2',0:'Correlation'}, inplace=True)
corrThresh = 0.90
strongCorrMatrix = strongCorrMatrix[((strongCorrMatrix['Correlation'] >= corrThresh) |
        (strongCorrMatrix['Correlation'] <= -corrThresh)) &
        (strongCorrMatrix['Pair1'] != strongCorrMatrix['Pair2'])]
strongCorrMatrix.reset_index(drop=True, inplace=True)

#Note 1 - Shpuld remove Attr7 and Attr14
same_column_check = df1[['Attr7','Attr14', 'Attr18']]

#Plot histogram of correlations for Target
sns.distplot(corrMat['Target'][corrMat['Target'] != 1],bins=10,kde=False,norm_hist=False)
plt.ylim(0,25)
plt.xlabel('Correlation')
plt.ylabel('Frequency of Occurrence')
plt.title('2009 Correlations of Attributes')
plt.savefig('Correlations_Target_2009.jpg')

#Top 10 most corelated attributes with Target
top_target_corr =abs(corrMat['Target']).nlargest(20)