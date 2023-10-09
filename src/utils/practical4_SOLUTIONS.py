import pandas as pd
import matplotlib as plt
import typing
import random

def dict_todataframe(test_dict):
    return pd.DataFrame.from_dict(test_dict)

stats = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

def get_column(test_df, column_name):
    return test_df[column_name]

def population_density(test_df):
    new_df = test_df.copy()
    new_df['density'] = new_df.population/new_df.area
    return new_df

def plot_population(test_df):
    new_df = test_df.copy()
    plt.bar(new_df.country, new_df.population)
    plt.title('Population of Countries')
    plt.ylabel('Country')
    plt.xlabel('Population (millions)')

def plot_area(test_df):
    new_df = test_df.copy()
    plt.pie(new_df.area, labels=new_df.country)

def csv_to_df(input_csv):
    df_data = pd.read_csv(input_csv)
    return df_data

def capitalize_columns(input_df):
    new_df = input_df.copy()
    for i in new_df.columns:
        if 'e' in i:
            new_df.rename(columns = {i:i.upper()}, inplace = True)
        else:
            pass
    return new_df

def math_passed_count(input_df):
    new_df = input_df.copy()
    c = 0
    for i in new_df['math score']:
        if i >= 50:
            c += 1
        else:
            continue
    return c

def did_pre_course(input_df):
    new_df = input_df.copy()
    new_df = new_df[new_df['test preparation course']=='completed']
    return new_df

def average_scores(input_df):
    new_df = input_df.copy()
    df = new_df.groupby(['parental level of education'])[['math score', 'reading score', 'writing score']].mean()
    return df

def add_age(input_df):
    new_df = input_df.copy()
    random.seed(42)
    new_df['age'] = pd.Series([random.randint(18, 66) for i in range(0, len(new_df))])
    return new_df

def female_top_score(input_df):
    new_df = input_df.copy()
    new_df['sum_score'] = new_df['math score'] + new_df['reading score'] + new_df['writing score']
    first_row = new_df[new_df['gender'] == 'female'].sort_values(by=['sum_score'], ascending=False).iloc[:1]
    return (first_row['math score'].iloc[0], first_row['reading score'].iloc[0], first_row['writing score'].iloc[0])

def add_grade(input_df):
    new_df = input_df.copy()
    new_df['percent'] = ((new_df['math score']+new_df['reading score']+new_df['writing score'])/300)*100
    def conditions(s):
        if s < 50:
            return 1
        elif s <= 65:
            return 2
        elif s <= 80:
            return 3
        elif s <= 90:
            return 4
        else:
            return 5
    new_df['grade'] = new_df.percent.apply(conditions, axis=1)
    return new_df

def math_bar_plot(input_df):
    new_df = input_df.copy()
    new_df = pd.DataFrame(new_df.groupby(['gender'], as_index=False)['math score'].mean())
    plt.bar(new_df['gender'], new_df['math score'])
    plt.title('Average Math Score by Gender')
    plt.ylabel('Gender')
    plt.xlabel('Math Score')

def writing_hist(input_df):
    new_df = input_df.copy()
    plt.hist(new_df['writing score'])
    plt.title('Distribution of Writing Scores')
    plt.xlabel('Writing Score')
    plt.ylabel('Number of Students')

def ethnicity_pie_chart(input_df):
    new_df = input_df.copy()
    asd = new_df.groupby(['race/ethnicity'], as_index=False).size()
    asd['eth_pct'] = (asd['size'] / len(new_df)) * 100
    plt.pie(asd.eth_pct, labels=asd['race/ethnicity'])
    plt.title('Proportion of Students by Race/Ethnicity')
