# Exno:1
# 1) Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Program
# For Data_set
```
import pandas as pd
df=pd.read_csv("Data_set.csv")
df
df.head()
df.tail()
df.isnull()
df.isnull().sum()
df.describe()
df.dropna(axis=1)
df.dropna(axis=0)
df.fillna(0)

```
 # Output
 # For Data_Set
 ![Alt text](<Screenshot 2024-03-05 141623.png>)
![Alt text](<Screenshot 2024-03-05 141758.png>)
![Alt text](<Screenshot 2024-03-05 141817.png>)
![Alt text](<Screenshot 2024-03-05 141846.png>)
![Alt text](<Screenshot 2024-03-05 141856.png>)
![Alt text](<Screenshot 2024-03-05 141910.png>)
![Alt text](<Screenshot 2024-03-05 141923.png>)
![Alt text](<Screenshot 2024-03-05 142009.png>)
![Alt text](<Screenshot 2024-03-05 142025.png>)


# 2) Outlier Detection and Removal

# AIM:
To read a given dataset and remove outliers and save a new dataframe.

# ALGORITHM:
(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (a) Using IQR detect weight outliers and print them

    (b) Using ZScore, detect height outliers and print them

# PROGRAM
# IQR:
```
import pandas as pd
import seaborn as sns
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
af
sns.boxplot(data=af)
sns.scatterplot(data=af)
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
iqr=af.quantile(0.5)
low=q1-1.5*iqr
low
high=q3+1.5*iqr
high
aq=af[((af>=low)&(af<=high))]
aq.dropna()
sns.boxplot(data=aq)
sns.scatterplot(data=aq)
```
# ZSCORE:
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
data = {'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}
df=pd.DataFrame(data)
df
sns.boxplot(data=df)
z = np.abs(stats.zscore(df))
print(df[z['weight']>3])
val =[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]
out=[]
def d_o(val):
    ts=3
    m=np.mean(val)
    sd=np.std(val)
    for i in val:
        z=(i-m)/sd
        if np.abs(z)>ts:
            out.append(i)
        return out
op = d_o(val)
op
```
# OUTPUT
# IQR

![Alt text](image-1.png)
![Alt text](2.png)
![Alt text](3.png)
![Alt text](4.png)
![Alt text](5.png)
![Alt text](6.png)
![Alt text](7.png)
![Alt text](8.png)
![Alt text](9.png)





# Z SCORE
![Alt text](image-15.png)
![Alt text](image-16.png)
![Alt text](image-18.png)
![Alt text](image-19.png)

# Result
Thus, the given data is read, cleansed and the cleaned data is saved into the file and the given data is read,remove outliers and save a new dataframe was created and executed successfully.

