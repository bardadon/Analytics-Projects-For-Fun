# A Visual History of Nobel Prize Winners

## Context

Between 1901 and 2016, the Nobel Prizes and the Prize in Economic Sciences were awarded 579 times to 911 people and organizations. The Nobel Prize is an international award administered by the Nobel Foundation in Stockholm, Sweden, and based on the fortune of Alfred Nobel, Swedish inventor and entrepreneur. In 1968, Sveriges Riksbank established The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel, founder of the Nobel Prize. Each Prize consists of a medal, a personal diploma, and a cash award.

A person or organization awarded the Nobel Prize is called Nobel Laureate. The word "laureate" refers to being signified by the laurel wreath. In ancient Greece, laurel wreaths were awarded to victors as a sign of honor.

## Project Tasks

    1. Who gets the Nobel Prize? 
    2. USA dominance
    3. The first woman to win the Nobel Prize
    4. Repeat laureates
    5. How old are you when you get the prize?
    6. Oldest and youngest winners

## The Data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('ggplot')
sns.set_context('notebook')

```


```python
data = pd.read_csv('archive.csv')
data.head()
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
      <th>Year</th>
      <th>Category</th>
      <th>Prize</th>
      <th>Motivation</th>
      <th>Prize Share</th>
      <th>Laureate ID</th>
      <th>Laureate Type</th>
      <th>Full Name</th>
      <th>Birth Date</th>
      <th>Birth City</th>
      <th>Birth Country</th>
      <th>Sex</th>
      <th>Organization Name</th>
      <th>Organization City</th>
      <th>Organization Country</th>
      <th>Death Date</th>
      <th>Death City</th>
      <th>Death Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1901</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 1901</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/1</td>
      <td>160</td>
      <td>Individual</td>
      <td>Jacobus Henricus van 't Hoff</td>
      <td>1852-08-30</td>
      <td>Rotterdam</td>
      <td>Netherlands</td>
      <td>Male</td>
      <td>Berlin University</td>
      <td>Berlin</td>
      <td>Germany</td>
      <td>1911-03-01</td>
      <td>Berlin</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1901</td>
      <td>Literature</td>
      <td>The Nobel Prize in Literature 1901</td>
      <td>"in special recognition of his poetic composit...</td>
      <td>1/1</td>
      <td>569</td>
      <td>Individual</td>
      <td>Sully Prudhomme</td>
      <td>1839-03-16</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1907-09-07</td>
      <td>Châtenay</td>
      <td>France</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1901</td>
      <td>Medicine</td>
      <td>The Nobel Prize in Physiology or Medicine 1901</td>
      <td>"for his work on serum therapy, especially its...</td>
      <td>1/1</td>
      <td>293</td>
      <td>Individual</td>
      <td>Emil Adolf von Behring</td>
      <td>1854-03-15</td>
      <td>Hansdorf (Lawice)</td>
      <td>Prussia (Poland)</td>
      <td>Male</td>
      <td>Marburg University</td>
      <td>Marburg</td>
      <td>Germany</td>
      <td>1917-03-31</td>
      <td>Marburg</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>462</td>
      <td>Individual</td>
      <td>Jean Henry Dunant</td>
      <td>1828-05-08</td>
      <td>Geneva</td>
      <td>Switzerland</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1910-10-30</td>
      <td>Heiden</td>
      <td>Switzerland</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1901</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 1901</td>
      <td>NaN</td>
      <td>1/2</td>
      <td>463</td>
      <td>Individual</td>
      <td>Frédéric Passy</td>
      <td>1822-05-20</td>
      <td>Paris</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1912-06-12</td>
      <td>Paris</td>
      <td>France</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The data base consists of {} rows and {} columns'.format(data.shape[0], data.shape[1]))   
```

    The data base consists of 969 rows and 18 columns
    

### Task 1 - Who gets the Nobel Prize? 
- Lets try to characterise the type of person who wins a Nobel Prize

#### Male vs Female


```python
male_female_ratio = data['Sex'].value_counts()
print('Since 1901 there were {} male winners and {} female Noble prize winners'.format(male_female_ratio[0],male_female_ratio[1]))
```

    Since 1901 there were 893 male winners and 50 female Noble prize winners
    


```python
g = sns.catplot(data = data,x = 'Sex', kind = 'count')
plt.xlabel('Sex', size = 18)
plt.ylabel('Count', size = 18)
plt.title('Gender Distriubtion of Nobel Prize Winners', size = 15)
plt.show()
```


    
![png](output_10_0.png)
    



```python
male_female_proporation = male_female_ratio[0] / male_female_ratio[1]
male_female_proporation
```




    17.86



#### Notes
- So __men won__ the Nobel Prize almost __18 times as much as women did__.
- Since women did not have access to education as much as men did at the time the Nobel Prize was first established __this is to be expected__.
- __Lets see if the gap between men and women has(hopefully) diminished over the years__


```python
male_vs_year = data[data['Sex'] == 'Male']['Year'].reset_index()
female_vs_year = data[data['Sex'] == 'Female']['Year'].reset_index()
```


```python
fig, ax = plt.subplots(2,1, sharex = True, figsize = [15,4])

ax[0].plot(male_vs_year['Year'],  male_vs_year.index, label = 'Male', c = 'r', linestyle = '--')
ax[0].set_ylabel('Count', size = 16)
ax[0].legend()

ax[1].plot(female_vs_year['Year'],  female_vs_year.index, label = 'Female', c = 'b', linestyle = '--' )
ax[1].set_xlabel('Year', size = 16)
ax[1].set_ylabel('Count', size = 16)
ax[1].legend()

fig.suptitle('Male vs Female Noble Prize Winners', size = 15)
plt.show()
```


    
![png](output_14_0.png)
    


- As we can see the amount of male nobel prize winners has been same over the years.
- in the last 30 years theres been more and more female nobel prize winners.
- Lets zoom in on those last 30 years


```python
last_30_df = data[data['Year'] > 1990]
male_vs_year_last_30 = last_30_df[last_30_df['Sex'] == 'Male']['Year'].reset_index()
female_vs_year_last_30 = last_30_df[last_30_df['Sex'] == 'Female']['Year'].reset_index()
```


```python
last_30_male_female_ratio = last_30_df['Sex'].value_counts()
print('Since 1990 there were {} male winners and {} female Noble prize winners'.format(last_30_male_female_ratio[0],last_30_male_female_ratio[1]))
```

    Since 1990 there were 297 male winners and 27 female Noble prize winners
    


```python
g = sns.catplot(data = last_30_df,x = 'Sex', kind = 'count')
plt.xlabel('Sex', size = 18)
plt.ylabel('Count', size = 18)
plt.title('Gender Distriubtion of Nobel Prize Winners - last 30 years', size = 16)
plt.show()
```


    
![png](output_18_0.png)
    



```python
male_female_proporation = last_30_male_female_ratio[0] / last_30_male_female_ratio[1]
male_female_proporation
```




    11.0



- So in the last 30 years the proportions of male vs female nobel prize winners is 11 compared to 18 for last century.
- As expected, the gap between male vs female is getting smaller over the years


### Country of Origin


```python
winners_by_country = data['Birth Country'].value_counts().head(20)
winners_by_country
```




    United States of America    276
    United Kingdom               88
    Germany                      70
    France                       53
    Sweden                       30
    Japan                        29
    Russia                       20
    Netherlands                  19
    Canada                       18
    Italy                        18
    Switzerland                  17
    Austria                      15
    Norway                       13
    China                        12
    Australia                    11
    Denmark                      11
    Germany (Poland)             10
    South Africa                  9
    Scotland                      9
    Belgium                       9
    Name: Birth Country, dtype: int64




```python
fig, ax = plt.subplots(figsize = [15,4])
ax.bar(x = winners_by_country.index, height = winners_by_country)
plt.xticks(rotation = 90, size = 12)
plt.xlabel('Country of Origin', size = 18)
plt.ylabel('Count', size = 18)
plt.title('Country of Origin Distribution', size = 18)
plt.show()
```


    
![png](output_23_0.png)
    


The majority of Nobel prize winners are either european(most likely from: Uk, France or Germany) or from the US.

### Category


```python
winners_by_category = data['Category'].value_counts().reset_index()
winners_by_category
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
      <th>index</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Medicine</td>
      <td>227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Physics</td>
      <td>222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chemistry</td>
      <td>194</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Peace</td>
      <td>130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Literature</td>
      <td>113</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Economics</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data = winners_by_category, x = 'index', y= winners_by_category.Category, palette='Blues_r')
plt.xlabel('Category', size = 16)
plt.ylabel('Count', size = 16)
plt.title('Nobel Prize Winner Distribution by Category', size = 18)
plt.show()
```


    
![png](output_27_0.png)
    


- It turns out that the most of the Nobel prizes are for the categories: Medicine, Physics and Chemistry. 

#### Summary
From this quick exploration, it turns out that the __average Noble prize winner__ is most probably a __US/European__ __Male__ that has studied __Medicine, Physics or Chemistry__.

### Task 2 - USA dominance
- As expected, USA is the dominant country when it comes to Noble prize winners.
- Lets check When did the USA start to dominate the Nobel Prize charts?


```python
us_winners_df = pd.DataFrame()
data['US_Winners'] = data['Birth Country'] == 'United States of America'
data['Decade'] = (np.floor(data['Year'] / 10)*10).astype('int')
prop_usa_winners = (data.groupby('Decade', as_index=False)['US_Winners'].mean()) 
prop_usa_winners
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
      <th>Decade</th>
      <th>US_Winners</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1900</td>
      <td>0.017241</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1910</td>
      <td>0.075000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1920</td>
      <td>0.074074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1930</td>
      <td>0.241935</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1940</td>
      <td>0.288889</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1950</td>
      <td>0.278481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1960</td>
      <td>0.262500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1970</td>
      <td>0.308411</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1980</td>
      <td>0.313131</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1990</td>
      <td>0.392523</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2000</td>
      <td>0.414286</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2010</td>
      <td>0.336735</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.subplots(figsize = [15,4])
plt.plot(prop_usa_winners.Decade, prop_usa_winners.US_Winners * 100)
plt.xlabel('Year', size = 16)
plt.ylabel('Proportion(%)', size = 16)
plt.title('Proportions of Us Winners', size = 18)
plt.show()
```


    
![png](output_32_0.png)
    


- It appears that at around the years 1920-1930 the US has started to become one of the dominant countries when it comes to noble prizes

### Task 3 - The first woman to win the Nobel Prize


```python
female_winners_df = data[data['Sex'] == 'Female'].sort_values('Year',ascending = True)
```


```python
print('\033[4m\033[1m Report - First Female Noble Prize Winner \033[0m\n\n{}'.format(female_winners_df.iloc[0,0:12]))
```

    [4m[1m Report - First Female Noble Prize Winner [0m
    
    Year                                                          1903
    Category                                                   Physics
    Prize                              The Nobel Prize in Physics 1903
    Motivation       "in recognition of the extraordinary services ...
    Prize Share                                                    1/4
    Laureate ID                                                      6
    Laureate Type                                           Individual
    Full Name                              Marie Curie, née Sklodowska
    Birth Date                                              1867-11-07
    Birth City                                                  Warsaw
    Birth Country                              Russian Empire (Poland)
    Sex                                                         Female
    Name: 19, dtype: object
    

- The first Female Noble Prize winner was Marie Curie in 1903 for her work in Physics.

### Task 4 - Repeat laureates
- Lets find those extraordinary people who have won the noble prize __more than once__


```python
import pandasql as ps
```


```python
def sql(query):
    return ps.sqldf(query)
```


```python
query = '''
SELECT *
FROM data
GROUP BY "Laureate ID"
HAVING COUNT("Laureate ID") > 1'''

repeat_laureates_df = sql(query)
repeat_laureates_df
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
      <th>Year</th>
      <th>Category</th>
      <th>Prize</th>
      <th>Motivation</th>
      <th>Prize Share</th>
      <th>Laureate ID</th>
      <th>Laureate Type</th>
      <th>Full Name</th>
      <th>Birth Date</th>
      <th>Birth City</th>
      <th>Birth Country</th>
      <th>Sex</th>
      <th>Organization Name</th>
      <th>Organization City</th>
      <th>Organization Country</th>
      <th>Death Date</th>
      <th>Death City</th>
      <th>Death Country</th>
      <th>US_Winners</th>
      <th>Decade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1903</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1903</td>
      <td>"in recognition of the extraordinary services ...</td>
      <td>1/4</td>
      <td>6</td>
      <td>Individual</td>
      <td>Marie Curie, née Sklodowska</td>
      <td>1867-11-07</td>
      <td>Warsaw</td>
      <td>Russian Empire (Poland)</td>
      <td>Female</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>1934-07-04</td>
      <td>Sallanches</td>
      <td>France</td>
      <td>0</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1949</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1949</td>
      <td>"for his prediction of the existence of mesons...</td>
      <td>1/1</td>
      <td>54</td>
      <td>Individual</td>
      <td>Hideki Yukawa</td>
      <td>1907-01-23</td>
      <td>Tokyo</td>
      <td>Japan</td>
      <td>Male</td>
      <td>Kyoto Imperial University</td>
      <td>Kyoto</td>
      <td>Japan</td>
      <td>1981-09-08</td>
      <td>Kyoto</td>
      <td>Japan</td>
      <td>0</td>
      <td>1940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1954</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1954</td>
      <td>"for the coincidence method and his discoverie...</td>
      <td>1/2</td>
      <td>62</td>
      <td>Individual</td>
      <td>Walther Bothe</td>
      <td>1891-01-08</td>
      <td>Oranienburg</td>
      <td>Germany</td>
      <td>Male</td>
      <td>University of Heidelberg</td>
      <td>Heidelberg</td>
      <td>Federal Republic of Germany</td>
      <td>1957-02-08</td>
      <td>Heidelberg</td>
      <td>West Germany (Germany)</td>
      <td>0</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1956</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1956</td>
      <td>"for their researches on semiconductors and th...</td>
      <td>1/3</td>
      <td>66</td>
      <td>Individual</td>
      <td>John Bardeen</td>
      <td>1908-05-23</td>
      <td>Madison, WI</td>
      <td>United States of America</td>
      <td>Male</td>
      <td>University of Illinois</td>
      <td>Urbana, IL</td>
      <td>United States of America</td>
      <td>1991-01-30</td>
      <td>Boston, MA</td>
      <td>United States of America</td>
      <td>1</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1958</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 1958</td>
      <td>"for the discovery and the interpretation of t...</td>
      <td>1/3</td>
      <td>71</td>
      <td>Individual</td>
      <td>Igor Yevgenyevich Tamm</td>
      <td>1895-07-08</td>
      <td>Vladivostok</td>
      <td>Russia</td>
      <td>Male</td>
      <td>University of Moscow</td>
      <td>Moscow</td>
      <td>Union of Soviet Socialist Republics</td>
      <td>1971-04-12</td>
      <td>Moscow</td>
      <td>Union of Soviet Socialist Republics (Russia)</td>
      <td>0</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2013</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 2013</td>
      <td>"for the development of multiscale models for ...</td>
      <td>1/3</td>
      <td>889</td>
      <td>Individual</td>
      <td>Martin Karplus</td>
      <td>1930-03-15</td>
      <td>Vienna</td>
      <td>Austria</td>
      <td>Male</td>
      <td>Université de Strasbourg</td>
      <td>Strasbourg</td>
      <td>France</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2014</td>
      <td>Physics</td>
      <td>The Nobel Prize in Physics 2014</td>
      <td>"for the invention of efficient blue light-emi...</td>
      <td>1/3</td>
      <td>906</td>
      <td>Individual</td>
      <td>Isamu Akasaki</td>
      <td>1929-01-30</td>
      <td>Chiran</td>
      <td>Japan</td>
      <td>Male</td>
      <td>Meijo University</td>
      <td>Nagoya</td>
      <td>Japan</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2014</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 2014</td>
      <td>"for the development of super-resolved fluores...</td>
      <td>1/3</td>
      <td>910</td>
      <td>Individual</td>
      <td>Stefan W. Hell</td>
      <td>1962-12-23</td>
      <td>Arad</td>
      <td>Romania</td>
      <td>Male</td>
      <td>Max Planck Institute for Biophysical Chemistry</td>
      <td>Göttingen</td>
      <td>Germany</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2015</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 2015</td>
      <td>"for mechanistic studies of DNA repair"</td>
      <td>1/3</td>
      <td>921</td>
      <td>Individual</td>
      <td>Tomas Lindahl</td>
      <td>1938-01-28</td>
      <td>Stockholm</td>
      <td>Sweden</td>
      <td>Male</td>
      <td>Francis Crick Institute</td>
      <td>Hertfordshire</td>
      <td>United Kingdom</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2015</td>
      <td>Chemistry</td>
      <td>The Nobel Prize in Chemistry 2015</td>
      <td>"for mechanistic studies of DNA repair"</td>
      <td>1/3</td>
      <td>922</td>
      <td>Individual</td>
      <td>Paul Modrich</td>
      <td>1946-06-13</td>
      <td>Raton, NM</td>
      <td>United States of America</td>
      <td>Male</td>
      <td>Howard Hughes Medical Institute</td>
      <td>Durham, NC</td>
      <td>United States of America</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 20 columns</p>
</div>



### Task 5 - How old are you when you get the prize?

- Lets check what is the average age of nobel prize winners


```python
query = '''
SELECT ("Year" - SUBSTR("Birth Date", 0, 5)) as "Age"
FROM data
'''
df = sql(query)
```


```python
data['Age'] = df['Age']
```


```python
avg_age = round(data['Age'].mean(),2)
print('The Average Age of a Noble Prize Winner is: {}'.format(avg_age)) 
```

    The Average Age of a Noble Prize Winner is: 59.49
    


```python
sns.histplot(data['Age'], kde = True)
plt.xlabel('Age', size = 16)
plt.ylabel('Count', size = 16)
plt.title('Noble Prize Winners - Age Distribution', size = 18)
plt.text(avg_age-3, 0,'Average Age\n{}'.format(avg_age), c = 'r')
plt.show()
```


    
![png](output_47_0.png)
    


- Lets also check the age distribution over the year.


```python
g = sns.lmplot(data = data, x = 'Year', y = 'Age', hue = 'Sex', aspect = 2)
g = g.set_axis_labels('Year', 'Age', size = 16)
plt.title('Age of Noble Prize Winners - Men vs Women', size = 18)
plt.show()
```


    
![png](output_49_0.png)
    


- Same for Category


```python
g = sns.lmplot(data = data, x = 'Year', y = 'Age', row = 'Category', aspect = 2,hue = 'Category' )
g = g.set_axis_labels('Year', 'Age', size = 16)
g.fig.suptitle('Age of Noble Prize Winners', size = 18)
plt.show()
```


    
![png](output_51_0.png)
    


### Task 6 - Oldest and youngest winners

#### Oldest


```python
# The oldest winner of a Nobel Prize as of 2016
oldest_winner = data.nlargest(1, 'Age')
oldest_winner
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
      <th>Year</th>
      <th>Category</th>
      <th>Prize</th>
      <th>Motivation</th>
      <th>Prize Share</th>
      <th>Laureate ID</th>
      <th>Laureate Type</th>
      <th>Full Name</th>
      <th>Birth Date</th>
      <th>Birth City</th>
      <th>...</th>
      <th>Sex</th>
      <th>Organization Name</th>
      <th>Organization City</th>
      <th>Organization Country</th>
      <th>Death Date</th>
      <th>Death City</th>
      <th>Death Country</th>
      <th>US_Winners</th>
      <th>Decade</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>825</th>
      <td>2007</td>
      <td>Economics</td>
      <td>The Sveriges Riksbank Prize in Economic Scienc...</td>
      <td>"for having laid the foundations of mechanism ...</td>
      <td>1/3</td>
      <td>820</td>
      <td>Individual</td>
      <td>Leonid Hurwicz</td>
      <td>1917-08-21</td>
      <td>Moscow</td>
      <td>...</td>
      <td>Male</td>
      <td>University of Minnesota</td>
      <td>Minneapolis, MN</td>
      <td>United States of America</td>
      <td>2008-06-24</td>
      <td>Minneapolis, MN</td>
      <td>United States of America</td>
      <td>False</td>
      <td>2000</td>
      <td>90.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
oldest_winner_name = data.nlargest(1, 'Age').loc[:,'Full Name'].values[0]
oldest_winner_age = data.nlargest(1, 'Age').loc[:,'Age'].values[0]
print(f'The Oldest Nobel Prize Winner is {oldest_winner_name!r} at the age of {oldest_winner_age!r}')
```

    The Oldest Nobel Prize Winner is 'Leonid Hurwicz' at the age of 90.0
    

#### Youngest


```python
# The youngest winner of a Nobel Prize as of 2016
data.nsmallest(1, 'Age')
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
      <th>Year</th>
      <th>Category</th>
      <th>Prize</th>
      <th>Motivation</th>
      <th>Prize Share</th>
      <th>Laureate ID</th>
      <th>Laureate Type</th>
      <th>Full Name</th>
      <th>Birth Date</th>
      <th>Birth City</th>
      <th>...</th>
      <th>Sex</th>
      <th>Organization Name</th>
      <th>Organization City</th>
      <th>Organization Country</th>
      <th>Death Date</th>
      <th>Death City</th>
      <th>Death Country</th>
      <th>US_Winners</th>
      <th>Decade</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>940</th>
      <td>2014</td>
      <td>Peace</td>
      <td>The Nobel Peace Prize 2014</td>
      <td>"for their struggle against the suppression of...</td>
      <td>1/2</td>
      <td>914</td>
      <td>Individual</td>
      <td>Malala Yousafzai</td>
      <td>1997-07-12</td>
      <td>Mingora</td>
      <td>...</td>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2010</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
youngest_winner_name = data.nsmallest(1, 'Age').loc[:,'Full Name'].values[0]
youngest_winner_age = data.nsmallest(1, 'Age').loc[:,'Age'].values[0]
print(f'The Youngest Nobel Prize Winner is {youngest_winner_name!r} at the age of {youngest_winner_age!r}')
```

    The Youngest Nobel Prize Winner is 'Malala Yousafzai' at the age of 17.0
    
