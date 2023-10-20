import pandas as pd
import typing
import matplotlib
import random
import src.utils
import src.weekly
#import str

#from pathlib import Path
#file_to_load = Path.cwd().parent.joinpath('data').joinpath('chipotle.tsv')

food = pd.read_csv(r'C:\DESKTOP\PythonEconRepoOnDesktop\ECOPY_23241\data\chipotle.tsv', sep='\t')

def change_price_to_float(input_df):
    input_df.item_price = input_df.item_price.str[1:]
    input_df.item_price = input_df.item_price.astype(float)
    return input_df

food = change_price_to_float(food)

def number_of_observations(input_df):
    return len(input_df)

def items_and_prices(input_df):
    df = input_df.copy()
    df = df.loc[:, ['item_name', 'item_price']]
    return df

def sorted_by_price(input_df):
    #df = change_price_to_float(input_df)
    input_df = input_df.sort_values(by=['item_price'], ascending=False)
    return input_df

def avg_price(input_df):
    return input_df.item_price.mean()

def unique_items_over_ten_dollars(input_df):
    #df = change_price_to_float(input_df)
    df = input_df[input_df.item_price>10]
    df = df.drop_duplicates(subset=["item_name", "item_price", "choice_description"])
    return df[["item_name", "choice_description", "item_price"]]

def items_starting_with_s(input_df):
    df = input_df.copy()
    df = df[df['item_name'].str.startswith('S')]
    #return df[['item_name']]
    return pd.Series(df[['item_name']].item_name.unique()).rename("item_name")

def first_three_columns(input_df):
    #df = change_price_to_float(input_df)
    return input_df.iloc[:, :3]

def every_column_except_last_two(input_df):
    #df = change_price_to_float(input_df)
    return input_df.iloc[:, :-2]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    result_df = input_df[columns_to_keep]
    result_df = result_df[input_df[column_to_filter].isin(rows_to_keep)]
    return result_df

def generate_quartile(input_df):
    #df = change_price_to_float(input_df)
    bins = [0, 9.99, 19.99, 29.99, float('inf')]
    labels = ['low-cost', 'medium-cost', 'high-cost', 'premium']
    input_df['Quartile'] = pd.cut(input_df['item_price'], bins=bins, labels=labels, right=False).astype(object)
    return pd.DataFrame(input_df)

def average_price_in_quartiles(input_df):
    df = generate_quartile(input_df)
    df2 = df.groupby(['Quartile'])['item_price'].mean()
    return df2

def minmaxmean_price_in_quartile(input_df):
    df = generate_quartile(input_df)
    df2 = df.groupby('Quartile').agg({'item_price': ['min', 'max', 'mean']})
    return df2


from  src.utils import distributions


def gen_uniform_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        # KUMULATÍV ÁTLAG (hibás) kumulatív összeghez jó lesz
        #a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        #sum = 0
        #cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
        #return cumsum
        # KUMULATÍV ÁTLAG
        a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        cumavg = pd.Series(a).expanding().mean()
        return cumavg.to_list()

    return [genList() for i in range(number_of_trajectories)]


def gen_logistic_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        # KUMULATÍV ÁTLAG (hibás) kumulatív összeghez jó lesz
        #a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        #sum = 0
        #cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
        #return cumsum
        # KUMULATÍV ÁTLAG
        a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        cumavg = pd.Series(a).expanding().mean()
        return cumavg.to_list()

    return [genList() for i in range(number_of_trajectories)]


def gen_laplace_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        # KUMULATÍV ÁTLAG (hibás) kumulatív összeghez jó lesz
        #a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        #sum = 0
        #cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
        #return cumsum
        # KUMULATÍV ÁTLAG
        a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        cumavg = pd.Series(a).expanding().mean()
        return cumavg.to_list()

    return [genList() for i in range(number_of_trajectories)]


def gen_cauchy_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        # KUMULATÍV ÁTLAG (hibás) kumulatív összeghez jó lesz
        #a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        #sum = 0
        #cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
        #return cumsum
        # KUMULATÍV ÁTLAG
        a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        cumavg = pd.Series(a).expanding().mean()
        return cumavg.to_list()

    return [genList() for i in range(number_of_trajectories)]


def gen_chi2_mean_trajectories(distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        # KUMULATÍV ÁTLAG (hibás) kumulatív összeghez jó lesz
        #a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        #sum = 0
        #cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
        #return cumsum
        # KUMULATÍV ÁTLAG
        a = [distribution.gen_rand() for i in range(length_of_trajectory)]
        cumavg = pd.Series(a).expanding().mean()
        return cumavg.to_list()

    return [genList() for i in range(number_of_trajectories)]