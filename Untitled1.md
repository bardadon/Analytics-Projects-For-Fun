# <ins>Exploration Project - Dr. Semmelweis and the Discovery of Handwashing<ins>
Analyzing the data that led to one of the most important discoveries of modern medicine: handwashing.

## Context
- The data shows the number of women giving birth from the year 1841 to 1846 at the Vienna General Hospital.
- Unfortunatly, giving birth at those times was very dangerous, and an alarming number of women died during childbirth due to what is known as __childbed fever__.

#### *From Wikipedia*
"Postpartum infections, also known as __childbed fever__ and puerperal fever, are any bacterial infections of the female reproductive tract following childbirth or miscarriage."

"puerperal fever, or childbed fever, was rampant, sometimes climbing to __40 percent of admitted patients__"

#### *About Dr. Ignaz Semmelweis*
- Dr. Ignaz Semmelweis is a Hungarian physician that worked in the Vienna General Hospital maternity clinic on a 3-year contract from 1846–1849.
- He was distrubed by the huge numbers of mortality rates, and after analyzing the mortality rates within the hospital, he was quite sure about the cause of it.
- He figured, that the __cause of the childbirth mortality was due to infections__.
- Eventually, he developed a theory of infections, in which he theorized that __decaying matter on the hands of doctors__ who were involed in autopsies, __are infecting the women__ at contact with the doctors during childbirth. 
- At June 1847 __Dr. Ignaz Semmelweis has implemented a handwashing policy__ for doctors in the matrnity clinic.
- In this project Ill try to explore the mortality rates __prior__ to Dr. Ignaz Semmelweis's handwash policy, and analyze the data that made Dr. Ignaz Semmelweis realize something is wrong with the procedures at Vienna General Hospital.

## Imports and Functions


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandasql as ps
sns.set_style('whitegrid')
from scipy import stats
from bs4 import BeautifulSoup
import requests
```


```python
def sql(query):
    '''
    Generates a Pandas Dataframe using SQL syntax.
    
    Parameters
    ----------
        arg_1 : SQL query(String)
    
    Returns
    -------
        Pandas DataFrame
    '''
    
    return ps.sqldf(query)


def ecdf(data):
    '''
    Compute ECDF for a one-dimensional array of measurements.
    
    Parameters
    ----------
        args_1 : 1D NumyPy Array or 1D Pandas Series
    
    Returns
    -------
        x, y variables for plotting ECDF
    
    '''
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def t_test(df1, df2, confidence_level = 0.05):
    '''
    Performing a Student's T-test.
    
    Parameters
    ----------
        args_1 : df1(Type: Pandas DataFrame)
        args_2 : df2(Type: Pandas DataFrame)
        args_3 : confidence_level(Type: Float, Default: 0.05)
    '''
    
    print("***Performing a Student's T-test***")
    
    statistic , p_value = stats.ttest_ind(df1, df2)
    print('\nThe P-Value for the T-test is: {:.5f}'.format(p_value[1]))
    if p_value[1] < confidence_level:
        print('Rejecting the Null Hypothesis - The Difference in Mean is Statistically Siginificant')
    else:
        print('Unable to Reject the Null Hypothesis - The Difference in Mean is NOT Statistically Siginificant')
```

## The Data


```python
# Loading the data
yearly_deaths = pd.read_csv('yearly_deaths_by_clinic.csv').drop('Unnamed: 0', axis = 'columns')
yearly_deaths
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>births</th>
      <th>deaths</th>
      <th>clinic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1841</td>
      <td>3036</td>
      <td>237</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1842</td>
      <td>3287</td>
      <td>518</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1843</td>
      <td>3060</td>
      <td>274</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1844</td>
      <td>3157</td>
      <td>260</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1845</td>
      <td>3492</td>
      <td>241</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1846</td>
      <td>4010</td>
      <td>459</td>
      <td>clinic 1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1841</td>
      <td>2442</td>
      <td>86</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1842</td>
      <td>2659</td>
      <td>202</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1843</td>
      <td>2739</td>
      <td>164</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1844</td>
      <td>2956</td>
      <td>68</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1845</td>
      <td>3241</td>
      <td>66</td>
      <td>clinic 2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1846</td>
      <td>3754</td>
      <td>105</td>
      <td>clinic 2</td>
    </tr>
  </tbody>
</table>
</div>



## Rate of death in each clinic

- The table above shows the mortatlity rates of two clinics in the hospital, lets look into the proporations of deaths in each clinic


```python
# Summing up the Number of Childbirths per clinic
births_by_clinic = pd.DataFrame(yearly_deaths.groupby('clinic')['births'].sum()).rename(columns = { 'births' :'Number of Childbirths'})
births_by_clinic
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of Childbirths</th>
    </tr>
    <tr>
      <th>clinic</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>clinic 1</th>
      <td>20042</td>
    </tr>
    <tr>
      <th>clinic 2</th>
      <td>17791</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Summing up the Number of Deaths per clinic
deaths_by_clinic = pd.DataFrame(yearly_deaths.groupby('clinic')['deaths'].sum()).rename(columns = {'deaths':'Number of Deaths'})
deaths_by_clinic
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number of Deaths</th>
    </tr>
    <tr>
      <th>clinic</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>clinic 1</th>
      <td>1989</td>
    </tr>
    <tr>
      <th>clinic 2</th>
      <td>691</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculating the proporations of death in clinic 1
clinic1_proportions = round(deaths_by_clinic.iloc[0,0] / births_by_clinic.iloc[0,0],2)
print('Rate of Death in Clinic 1 is: {}'.format(clinic1_proportions))
```

    Rate of Death in Clinic 1 is: 0.1
    


```python
# Calculating the proporations of death in clinic 1
clinic2_proportions = round(deaths_by_clinic.iloc[1,0] / births_by_clinic.iloc[1,0],2)
print('Rate of Death in Clinic 2 is: {}'.format(clinic2_proportions))
```

    Rate of Death in Clinic 2 is: 0.04
    

## Cumulative Plots


```python
# Using SQL syntax to Create a Dataframe for each of the clinics
query1 = '''SELECT year, deaths
            From yearly_deaths
            WHERE clinic = "clinic 1"
            '''
query2 = '''SELECT year, deaths
            From yearly_deaths
            WHERE clinic = "clinic 2"
            '''
clinic1_df = sql(query1)
clinic2_df = sql(query2)
```


```python
# Plotting a Cumulative Sum of all the deaths throughout the years 1841 to 1846
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(clinic1_df['year'], clinic1_df['deaths'].cumsum(), label = 'Clinic 1')
ax.plot(clinic2_df['year'], clinic2_df['deaths'].cumsum(), label = 'Clinic 2')
plt.xlabel('Year', size = 14)
plt.ylabel('Death Count', size = 14)
plt.title('Cumulative Sum of Death Count - Years 1841 to 1846', size = 18)
plt.legend()
plt.show()
```


    
![png](output_17_0.png)
    



```python
# Generating x and y variables to plot the Estimated Cumulative Distribution Function(ecdf)
x_clinic1, y_clinic1 = ecdf(clinic1_df.deaths)
x_clinic2, y_clinic2 = ecdf(clinic2_df.deaths)
```


```python
# Plotting the ecdf and annotating the 25th 50th and the 75th percentile of death count per clinic
fig, ax = plt.subplots(figsize = (15,6))
ax.plot(x_clinic1, y_clinic1, label = 'Clinic 1')
ax.plot(x_clinic2, y_clinic2, label = 'Clinic 2')
plt.xlabel('Death Count', size = 14)
plt.ylabel('Percentile', size = 14)
plt.title('Cumulative Distribution - Years 1841 to 1846', size = 18)
plt.annotate('Clinic 1 - 25th Percentile({:.0f} Deaths)'.format(np.quantile(clinic1_df.deaths, 0.25)), xy = (np.quantile(clinic1_df.deaths,0.25),0.25), color = 'r', size = 12)
plt.annotate('Clinic 2 - 25th Percentile({:.0f} Deaths)'.format(np.quantile(clinic2_df.deaths, 0.25)), xy = (np.quantile(clinic2_df.deaths, 0.25), 0.25), color = 'r', size = 12)
plt.annotate('Clinic 1 - 50th Percentile({:.0f} Deaths)'.format(np.median(clinic1_df.deaths)), xy = (np.median(clinic1_df.deaths),0.5), color = 'r', size = 12)
plt.annotate('Clinic 2 - 50th Percentile({:.0f} Deaths)'.format(np.median(clinic2_df.deaths)), xy = (np.median(clinic2_df.deaths), 0.5), color = 'r', size = 12)
plt.annotate('Clinic 1 - 75th Percentile({:.0f} Deaths)'.format(np.quantile(clinic1_df.deaths,0.75)), xy = (np.quantile(clinic1_df.deaths,0.75),0.75), color = 'r', size = 12)
plt.annotate('Clinic 2 - 75th Percentile({:.0f} Deaths)'.format(np.quantile(clinic2_df.deaths,0.75)), xy = (np.quantile(clinic2_df.deaths,0.75), 0.75), color = 'r', size = 12)
plt.legend()
plt.show()
```


    
![png](output_19_0.png)
    


#### Notes:
- As we can see, about __1 in 10 women died during childbirth in clinic 1, and about 1 in 25 in clinic 2__.
- Also just by looking at these plots we can tell that __there is a clear difference in mortality rates__ between the two clinics.
- This difference is one of the first things that led Dr. Ignaz Semmelweis to the policy of handwashing.


### ** Some domain Knowledge from wikipedia **
- It turns out, that clinic 1 was mostly served by medical students, while clinic 2 was mostly served midwife students.
- Also, the __clinic 1 medical students were also incharge of the autopsy rooms__ and they spent some of their time examining corepses.
- Dr. Semmelweis started to suspect that something on the corpses spread from the hands of the medical students, caused childbed fever.

## Hypothesis Testing
- Before jumping to conclusion lets perform an Hypothesis Test if the difference between the clinics is Significant.
- The test will be a Student's T-test due to the small sample size.
- Lets define:
    - __Null Hypothesis__: The mean clinic 1 is *equal* to clinic 2.
    - __Alternate Hypothesis__: The mean in clinic 1 is statistically *different* than clinic 2.
    - Required Confidence Level - 0.05


```python
# Performing a Student's T-test 
t_test(clinic1_df, clinic2_df)
```

    ***Performing a Student's T-test***
    
    The P-Value for the T-test is: 0.00294
    Rejecting the Null Hypothesis - The Difference in Mean is Statistically Siginificant
    

#### Notes:
- So it turns out that Dr. Ignaz Semmelweis was correct!
- The two clinics are different after all.
- At this point in time, Dr. Ignaz Semmelweis in a desperate attempt to stop the high mortality rates, decreed: *Wash your hands!*
- This was an unorthodox and controversial request, nobody in Vienna knew about bacteria at this point in time. 
- __Let's import the monthly data of Clinic 1 from Wikipedia to see if the handwashing had any effect.__

## Web Scraping from Wiki - Monthly Data


```python
# Setting the required URL address 
wiki_url = 'https://en.wikipedia.org/wiki/Historical_mortality_rates_of_puerperal_fever'
table_class="wikitable sortable jquery-tablesorter"

# Sending to Wikipedia a GET request
response = requests.get(wiki_url)

# Recieving the response code from Wikipedia
status_code = response.status_code

# Checking if Access is Authorized
if status_code == 200:
    print('Access Authorized')
else:
    print('No Access')
print()
```

    Access Authorized
    
    


```python
# Parsing the response in an HTML format
soup = BeautifulSoup(response.text, 'html.parser')

# Finding the required table from the Wikipedia page
table=soup.find('table',{'class':"wikitable"})
```


```python
# Reading table data
clinic1_monthly_df = pd.read_html(str(table))

# Converting list to dataframe
clinic1_monthly_df = pd.DataFrame(clinic1_monthly_df[0]).drop(columns = ['Year', 'Notes'], axis = 1)
clinic1_monthly_df.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Month</th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>January 1841</td>
      <td>254</td>
      <td>37</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>February 1841</td>
      <td>239</td>
      <td>18</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>March 1841</td>
      <td>277</td>
      <td>12</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>April 1841</td>
      <td>255</td>
      <td>4</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>May 1841</td>
      <td>255</td>
      <td>2</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>June 1841</td>
      <td>200</td>
      <td>10</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>July 1841</td>
      <td>190</td>
      <td>16</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>August 1841</td>
      <td>222</td>
      <td>3</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>September 1841</td>
      <td>213</td>
      <td>4</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>October 1841</td>
      <td>236</td>
      <td>26</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>November 1841</td>
      <td>235</td>
      <td>53</td>
      <td>22.6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>December 1841</td>
      <td>na</td>
      <td>na</td>
      <td>na</td>
    </tr>
  </tbody>
</table>
</div>



#### Notes:
- This data shows the monthly death rates of women before and after the handwashing policy was implemented.
- Lets see how much the policy affected the death rates.

## Analyzing data before and after the handwashing policy

### Some Pre-Processing First


```python
# Checking the feature's data type
print(clinic1_monthly_df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 99 entries, 0 to 98
    Data columns (total 4 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   Month     99 non-null     object
     1   Births    99 non-null     object
     2   Deaths    99 non-null     object
     3   Rate (%)  99 non-null     object
    dtypes: object(4)
    memory usage: 3.2+ KB
    None
    

- The features are of type "object", each feature needs to be changed to a more appropriate type.


```python
# Checking for missing values
clinic1_monthly_df.isna().sum()
```




    Month       0
    Births      0
    Deaths      0
    Rate (%)    0
    dtype: int64



- No missing values


```python
# Sorting the Birth feature
clinic1_monthly_df.Births.sort_values(ascending = False)
```




    11     na
    98    406
    96    403
    97    389
    95    373
         ... 
    5     200
    29    196
    31    193
    30    191
    6     190
    Name: Births, Length: 99, dtype: object



- Birth column has a "na" value, Ill need to get rid of that in order to change the data type.


```python
# Finding the index to drop
index_to_drop = clinic1_monthly_df[clinic1_monthly_df.Births.str.find('na') == 0].index[0]
index_to_drop
```




    11




```python
# Dropping the index
clinic1_monthly_df.drop(index=index_to_drop, inplace = True)
```


```python
# Changing each column to a more appropriate data type
clinic1_monthly_df['Births'] = clinic1_monthly_df.Births.astype('int')
clinic1_monthly_df['Deaths'] = clinic1_monthly_df.Deaths.astype('int')
clinic1_monthly_df['Rate (%)'] = clinic1_monthly_df['Rate (%)'].astype('float')
clinic1_monthly_df['Month'] = pd.to_datetime(clinic1_monthly_df.Month)
```


```python
# Checking the new data types
clinic1_monthly_df.dtypes
```




    Month       datetime64[ns]
    Births               int32
    Deaths               int32
    Rate (%)           float64
    dtype: object




```python
# Setting the Month datetime column as the index
clinic1_monthly_df = clinic1_monthly_df.set_index('Month')
```


```python
# New Dataframe look like this
clinic1_monthly_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1841-01-01</th>
      <td>254</td>
      <td>37</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>1841-02-01</th>
      <td>239</td>
      <td>18</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>1841-03-01</th>
      <td>277</td>
      <td>12</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>1841-04-01</th>
      <td>255</td>
      <td>4</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>1841-05-01</th>
      <td>255</td>
      <td>2</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1848-11-01</th>
      <td>310</td>
      <td>9</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>1848-12-01</th>
      <td>373</td>
      <td>5</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>1849-01-01</th>
      <td>403</td>
      <td>9</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>1849-02-01</th>
      <td>389</td>
      <td>12</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>1849-03-01</th>
      <td>406</td>
      <td>20</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
<p>98 rows × 3 columns</p>
</div>



### Plotting the effects of the handwashing policy
- According to Wiki, the handwashing policy started around: Mid 1847 


```python
number_of_deaths_pre_policy = clinic1_monthly_df[clinic1_monthly_df.index == '1847-04-01'].Deaths[0]

fig, ax = plt.subplots(figsize = (15,6))
ax.plot(clinic1_monthly_df.index, clinic1_monthly_df.Deaths)
plt.annotate('Handwashing Policy \nImplementation',xy = (pd.Timestamp('1847-04-01'),number_of_deaths_pre_policy), color = 'r',
                size = 14, arrowprops = {'arrowstyle':'->', 'color':'gray'}, xytext = (pd.Timestamp('1848-04-01'),50))

plt.show()
```


    
![png](output_45_0.png)
    


- Look like washing the new policy in clinic 1 almost immediately reduced the number of deaths.

### Dividing the data set to before and after the handwashing policy.



```python
pre_policy_df = clinic1_monthly_df[clinic1_monthly_df.index < '1847-05-01']
pre_policy_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1846-12-01</th>
      <td>298</td>
      <td>16</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>1847-01-01</th>
      <td>311</td>
      <td>10</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>1847-02-01</th>
      <td>312</td>
      <td>6</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>1847-03-01</th>
      <td>305</td>
      <td>11</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>1847-04-01</th>
      <td>312</td>
      <td>57</td>
      <td>18.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
after_policy_df = clinic1_monthly_df[clinic1_monthly_df.index >= '1847-05-01']
after_policy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
    <tr>
      <th>Month</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1847-05-01</th>
      <td>294</td>
      <td>36</td>
      <td>12.2</td>
    </tr>
    <tr>
      <th>1847-06-01</th>
      <td>268</td>
      <td>6</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>1847-07-01</th>
      <td>250</td>
      <td>3</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1847-08-01</th>
      <td>264</td>
      <td>5</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>1847-09-01</th>
      <td>262</td>
      <td>12</td>
      <td>4.6</td>
    </tr>
  </tbody>
</table>
</div>



### Analyzing the differences


```python
pre_policy_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>75.000000</td>
      <td>75.000000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>257.026667</td>
      <td>26.986667</td>
      <td>10.481333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>34.241974</td>
      <td>18.026627</td>
      <td>7.119989</td>
    </tr>
    <tr>
      <th>min</th>
      <td>190.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>236.500000</td>
      <td>11.500000</td>
      <td>4.400000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>254.000000</td>
      <td>26.000000</td>
      <td>10.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>278.500000</td>
      <td>40.000000</td>
      <td>15.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>336.000000</td>
      <td>75.000000</td>
      <td>31.400000</td>
    </tr>
  </tbody>
</table>
</div>




```python
after_policy_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Births</th>
      <th>Deaths</th>
      <th>Rate (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>299.521739</td>
      <td>7.739130</td>
      <td>2.547826</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47.997035</td>
      <td>7.846351</td>
      <td>2.573443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>246.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>266.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>283.000000</td>
      <td>6.000000</td>
      <td>2.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>311.000000</td>
      <td>10.500000</td>
      <td>3.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>406.000000</td>
      <td>36.000000</td>
      <td>12.200000</td>
    </tr>
  </tbody>
</table>
</div>



#### Notes:
- So Its clear that there are huge differences before and after the handwashing policy.
- __The average number of deaths post policy is about 3 times as lower!!!__
- Lets check out the distribution of Death counts pre and post policy


```python
# Checking out the Kernel Density Estimation of each Dataframe
fig, ax = plt.subplots(figsize = (15,6))
ax = sns.kdeplot(data = after_policy_df, x = after_policy_df.Deaths, shade = True, label = 'Post-Policy')
ax = sns.kdeplot(data = pre_policy_df, x = pre_policy_df.Deaths, shade =True, label = 'Pre-Policy')
plt.xlabel('Deaths', size = 16)
plt.ylabel('Density', size = 16)
plt.title('Distribution of Death Count - Post and Pre Handwashing Policy', size = 18)
plt.legend()
plt.show()
```


    
![png](output_54_0.png)
    


#### Notes:
- There is a clear difference, but again, before jumping to conclutions __lets perform an hypothesis test to see if the differences is significant.__
- This time ill use a __Bootstrap Analysis of the data__.
- The goal of the analysis is to resmaple the data and thus create replicates it.
- This will allow us to check whether or not __the difference in the average number of deaths between the two clinics, will remain different even if we were to take A LOT of samples from each clinic.__
- If so, we can declare that the difference in the two samples of data(pre and post handwashing policy), is __Statistically Significant__, and will prove that the policy did in fact worked and it was __NOT a matter of random chance__.


```python
# Creating 10000 samples of the pre and post policy data
pre_policy_bootstrap_sample = np.random.choice(pre_policy_df.Deaths, size = 10000)
post_policy_bootstrap_sample = np.random.choice(after_policy_df.Deaths, size = 10000)

# Calculating the difference in mean
mean_diff = round(np.mean(post_policy_bootstrap_sample) - np.mean(pre_policy_bootstrap_sample),5)
mean_diff
```




    -19.4068



- So it looks like the policy has decresed the mean number of deaths in about 19 deaths per month!!
- Lets calculate a 95% Confidence interval to see the full extent of the benefits of the handwashing policy.


```python
# A bootstrap analysis of the reduction of deaths due to handwashing
boot_mean_diff = []
for i in range(3000):
    boot_before = np.random.choice(pre_policy_df.Deaths, size = 10000)
    boot_after = np.random.choice(after_policy_df.Deaths, size = 10000)
    boot_mean_diff.append( np.mean(boot_after) - np.mean(boot_before) )


```


```python
# Calculating a 95% confidence interval from boot_mean_diff 
confidence_interval = pd.Series(boot_mean_diff).quantile([0.025,0.975])
confidence_interval
```




    0.025   -19.63495
    0.975   -18.87719
    dtype: float64



- In other words, I can say with a 95% confidence that after the handwashing policy, the average number of deaths will decrease by up to 18.87 to 19.63 per month!!!
- And its all thanks to Dr. Semmelweis and his Discovery of Handwashing.
