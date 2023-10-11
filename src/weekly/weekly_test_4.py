import pandas as pd
import typing
import matplotlib.pyplot as plt
import random
#import src.distributions
import src.weekly


euro12 = pd.read_csv('data/Euro_2012_stats_TEAM.csv')


def number_of_participants(input_df):
    df = input_df.copy()
    return len(df)

def goals(input_df):
    df = input_df.copy()
    return df.iloc[:, ['Team', 'Goals']]

def avg_goal(input_df):
    df = input_df.copy()
    return df.Goals.mean()

def countries_over_five(input_df):
    df = input_df.copy()
    return df[df['Goals']>=6]

def countries_starting_with_g(input_df):
    df = input_df.copy()
    gs = df[df['Team'].str.startswith('G')]
    return gs

def first_seven_columns(input_df):
    df = input_df.copy()
    return df.iloc[:, :7]


def every_column_except_last_three(input_df):
    df = input_df.copy()
    return df.iloc[:, :-3]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    df = input_df.copy()
    selected_columns = df[columns_to_keep]
    selected_rows = df[df[column_to_filter].isin(rows_to_keep)]
    return selected_columns.join(selected_rows)

def generate_quartile(input_df):
    new_df = input_df.copy()
    def conditions(s):
        if s <= 2:
            return 4
        elif s <= 4:
            return 3
        elif s <= 5:
            return 2
        else:
            return 1

    new_df['Quartile'] = new_df.Goals.apply(conditions, axis=1)
    return new_df

def average_yellow_in_quartiles(input_df):
    df = generate_quartile(input_df)
    quartiles = df.groupby('Quartile')['Passes'].mean()
    return quartiles

def minmax_block_in_quartile(input_df):
    df = generate_quartile(input_df)
    grouped = df.groupby('Quartile')
    min_max_blocks = grouped['Blocks'].agg(['min', 'max'])
    return min_max_blocks

def scatter_goals_shots(input_df):
    df = input_df.copy()
    goals = df['Goals']
    shots = df['Shots on target']

    plt.figure(figsize=(8, 6))
    plt.scatter(goals, shots, alpha=0.5)
    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.show()

    return plt.gcf()

def scatter_goals_shots_by_quartile(input_df):
    df = generate_quartile(input_df)
    goals = df['Goals']
    shots = df['Shots on target']

    q1 = df.loc[df['Quartile'] == 1]
    q2 = df.loc[df['Quartile'] == 2]
    q3 = df.loc[df['Quartile'] == 3]
    q4 = df.loc[df['Quartile'] == 4]

    plt.figure(figsize=(8, 6))
    plt.scatter(q1['Goals'], q1['Shots on target'], label='Quartile 1', alpha=0.5, c='b')
    plt.scatter(q2['Goals'], q2['Shots on target'], label='Quartile 2', alpha=0.5, c='g')
    plt.scatter(q3['Goals'], q3['Shots on target'], label='Quartile 3', alpha=0.5, c='r')
    plt.scatter(q4['Goals'], q4['Shots on target'], label='Quartile 4', alpha=0.5, c='m')
    plt.title('Goals and Shot on target')
    plt.xlabel('Goals')
    plt.ylabel('Shots on target')
    plt.legend(title='Quartiles')
    plt.show()

    return plt.gcf()

def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    def genList():
        try:
            a = [pareto_distribution.gen_rand() for i in range(length_of_trajectory)]
            sum = 0
            cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
            return cumsum
        except:
            a = [src.weekly.weakly_test_2.ParetoDistribution(1, 1).gen_rand() for i in range(length_of_trajectory)]
            sum = 0
            cumsum = [sum := (sum + s) / (a.index(s) + 1) for s in a]
            return cumsum

    return [genList() for i in range(number_of_trajectories)]