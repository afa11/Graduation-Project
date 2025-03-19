
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_columns(df, x_col, y_col, title="Scatter Plot", xlabel=None, ylabel=None):

    plt.figure(figsize=(10, 5))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')

    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_the_data_and_convert_datetime(path):

    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def drop_a_column(df, column_name):

    newdf = df.drop(column_name, axis='columns')
    return newdf


def plot_box(df, column, title="Box Plot"):

    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[column])
    
    plt.title(title)
    plt.ylabel(column)
    plt.grid(True)
    plt.show()

def plot_violin(df, column_name):
    sns.violinplot(x=column_name, data=df)



def filter_a_day(specific_date,df):

    
    df_filtered = df[df['timestamp'].dt.date == pd.to_datetime(specific_date).date()]

    df_filtered = df_filtered.reset_index(drop=True)

    return df_filtered


def aggregate_as_a_minute(df_day1):
    df_day1 = df_day1.set_index('timestamp')

    # Resample data by minute and compute mean
    df_minute_avg_day1 = df_day1.resample('T').mean().reset_index()

    return df_minute_avg_day1


def aggregate_with_sliding_window(df, window_size, slide):
    df = df.set_index('timestamp')

    # Resample data by minute and compute mean
    df_minute_avg = df.resample('T').mean()

    # Compute rolling mean with specified window size and step
    df_sliding_avg = df_minute_avg.rolling(window=window_size).mean()[::slide].dropna().reset_index()

    return df_sliding_avg


def aggregate_with_sliding_window_rowwise(df, window_size, slide):
    df = df.reset_index(drop=True)  # Satır indexlerini sıfırdan başlat
    
    rolling_means = [
        df.iloc[i:i+window_size].mean()
        for i in range(0, len(df) - window_size + 1, slide)
    ]

    df_sliding_avg = pd.DataFrame(rolling_means)
    
    return df_sliding_avg

def filter_rows_between_the_given_timestamps(df, start, end, varible_name="timestamp"):

    new_df = df.loc[(df[varible_name] >= start) & (df[varible_name] <= end)]
    return new_df


def change_the_values_by_applying_a_time_filter(df, start_date, end_date, feature, new_value):

    df_new = df.copy()

    df_new.loc[(df_new["timestamp"] <= end_date) & (df_new["timestamp"] >= start_date), feature] = new_value
    return df_new


def apply_ttest(df, column_names, variable):

    summary_stats = df.groupby('condition')[column_names].agg(['mean', 'median', 'std'])
    print(summary_stats[variable])

    condition_0 = df[df['condition'] == 0][variable]
    condition_1 = df[df['condition'] == 1][variable]

    print()

    t_stat, p_value = stats.ttest_ind(condition_0, condition_1)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference: {p_value < 0.05}")
    