
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


f1_start = "2020-04-18 00:00:00"
f1_finish = "2020-04-18 23:59:00"
f2_start = "2020-05-29 23:30:00"
f2_finish = "2020-05-30 06:00:00"
f3_start = "2020-06-05 10:00:00"
f3_finish = "2020-06-07 14:30:00"
f4_start = "2020-07-15 14:30:00"
f4_finish = "2020-07-15 19:00:00"


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



def plot_columns_failure_comparison(df, x_col, y_col, color_col, title="Scatter Plot", xlabel=None, ylabel=None):
    colors = {0: 'red', 1: 'blue', 2: 'black'}
    
    plt.figure(figsize=(15, 8))
    
    # Plot each category as a separate scatter plot
    for category in df[color_col].unique():
        subset = df[df[color_col] == category]
        plt.scatter(subset[x_col], subset[y_col], 
                   color=colors.get(category, 'gray'),
                   label=f'Category {category}',
                   alpha=0.7,  # Add some transparency
                   s=30)  # Control point size
    
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def counter_for_maintenance(df, start_of_interval, finish_date):

    start_of_interval = pd.to_datetime(start_of_interval)
    finish_date = pd.to_datetime(finish_date)
    

    df_new = df.copy()
    

    
    counter = 0
    

    for idx, row in df_new.iterrows():
        time = row["timestamp"]
        
        if time >= finish_date:
            break
        elif time < start_of_interval:
            pass
        else:
            counter += (1/(24*60*10))
            df_new.at[idx, "counter"] = counter
    
    return df_new


def scale_columns(df, columns):
    std_scaler = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[columns] = std_scaler.fit_transform(df[columns])  # Scale only numerical columns
    
    return df_scaled




def apply_kmeans_clustering(df, number_of_clusters, target_variable):

    X = df.select_dtypes(include=[np.number]).drop(columns=[target_variable], errors="ignore")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    return df 


def check_cluster_distribution(df, condition_column, cluster_column):

    distribution = pd.crosstab(df[cluster_column], df[condition_column])
    print("\nCluster Distribution by Condition:\n", distribution)

    return distribution


def apply_smote(df, target_column, seed):

    X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = y_resampled

    print("Before : ", df[target_column].value_counts())
    print("After : ", y_resampled.value_counts())

    return df_resampled


def apply_random_forest_and_get_results(df, target, seed=10):

    target = df[target]

    X = df.drop("condition", axis='columns')


    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.33, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print(pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False) )

    
    return model, accuracy


def adjust_datetime(date, direction, days):

    date_format = "%Y-%m-%d %H:%M:%S"
    date_obj = datetime.strptime(date, date_format)

    if direction =="forward":
        change = timedelta(days=days)
    else:
        change = -timedelta(days=days)

    new_date_obj = date_obj + change

    return new_date_obj.strftime(date_format)


"""def change_condition(df, n_rows=3): # TEKRAR YAZILACAK

    # Veri çerçevesinin bir kopyasını oluşturalım
    df_copy = df.copy()
    
    # 'condition' sütunundaki değişimleri bulalım
    # Bir önceki değer 0 ve şimdiki değer 1 olan yerleri tespit edelim
    condition_change = (df_copy['condition'].shift(1) == 0) & (df_copy['condition'] == 1)
    
    # Değişim noktalarının indekslerini bulalım
    change_indices = df_copy[condition_change].index.tolist()
    
    # Her değişimden önce n_rows kadar satırı 2 ile değiştirelim
    for idx in change_indices:
        # n_rows kadar geriye gidelim, ancak negatif indeks oluşmamasına dikkat edelim
        start_idx = max(0, idx - n_rows)
        
        # Sadece condition değeri 0 olan satırları 2 ile değiştirelim
        for i in range(start_idx, idx):
            if df_copy.loc[i, 'condition'] == 0:
                df_copy.loc[i, 'condition'] = 2
    
    return df_copy"""


def group_rows_by_condition(df, group_size=400): # tekrar yazılacak

    total_rows = len(df)

    num_groups = int(np.ceil(total_rows / group_size))
    
    result_data = []
    
    for i in range(num_groups):
        # Mevcut grubu alıyoruz
        start_idx = i * group_size
        end_idx = min((i + 1) * group_size, total_rows)
        current_group = df.iloc[start_idx:end_idx]



        
        # Gruptaki condition değerlerini kontrol ediyoruz
        if 1 in current_group['condition'].values:
            group_condition = 1
        else:
            group_condition = 0
        
        # Grubun ortalama proba değerini hesaplıyoruz
        group_proba = current_group['proba'].mean()
        
        # Yeni satırı sonuç listesine ekliyoruz
        result_data.append({
            'group_id': i,
            'start_row': start_idx,
            'end_row': end_idx - 1,
            'row_count': end_idx - start_idx,
            'proba': group_proba,
            'condition': group_condition
        })
    
    # Sonuç listesini DataFrame'e dönüştürüyoruz
    result_df = pd.DataFrame(result_data)
    
    return result_df


"""
def get_the_probabilities_with_logistic_regression(df, n1, n2, n3, n4, n5, n6, n7, n8, printt):

    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))

    df_log_reg_train = pd.concat([df1, df2, df3], ignore_index=True).copy()
    df_log_reg_test = df4.copy()

    y_train = df_log_reg_train["condition"]
    X_train = df_log_reg_train.drop(["condition", "timestamp"], axis=1)

    y_test = df_log_reg_test["condition"]
    X_test = df_log_reg_test.drop(["condition", "timestamp"], axis=1)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    feature_names = X_train.columns
    coef_df = pd.DataFrame(model.coef_[0], index=feature_names, columns=['Coefficient'])
    coef_df_sorted = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

    if printt == "yes":

        print(coef_df_sorted)
        print("Intercept:", model.intercept_[0])

    y_proba = model.predict_proba(X_test)[:, 1]

    return y_proba, y_test"""






def get_the_probabilities_with_logistic_regressionn(df, n1, n2, n3, n4, n5, n6, n7, n8, printt):

    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))
    
    df_log_reg_train = pd.concat([df1, df2, df3], ignore_index=True).copy()
    df_log_reg_test = df4.copy()
    
    y_train = df_log_reg_train["condition"]
    X_train = df_log_reg_train.drop(["condition", "timestamp"], axis=1)
    
    y_test = df_log_reg_test["condition"]
    X_test = df_log_reg_test.drop(["condition", "timestamp"], axis=1)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Calculate p-values
    from scipy import stats
    
    # Get the standard errors
    predictions = model.predict(X_train)
    residuals = y_train - predictions
    
    # Calculate variance residual and standard error
    mse = np.sum(residuals**2) / (len(y_train) - X_train.shape[1] - 1)
    var_coef = mse * np.linalg.inv(np.dot(X_train.T, X_train)).diagonal()
    se_coef = np.sqrt(var_coef)
    
    # Calculate z-scores and p-values
    z_scores = model.coef_[0] / se_coef
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_scores]
    
    # Calculate R-squared
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_train)
    r_squared = r2_score(y_train, y_pred)
    
    # Calculate F-value
    from sklearn.metrics import mean_squared_error
    mse_model = mean_squared_error(y_train, y_pred)
    mse_baseline = np.var(y_train)
    f_value = (mse_baseline - mse_model) / mse_model * (len(y_train) - X_train.shape[1] - 1) / X_train.shape[1]
    
    feature_names = X_train.columns
    coef_df = pd.DataFrame({
        'Coefficient': model.coef_[0],
        'p_value': p_values
    }, index=feature_names)
    
    coef_df_sorted = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    if printt == "yes":
        print(coef_df_sorted)
        print("Intercept:", model.intercept_[0])
        print("R-squareddd:", r_squared)
        print("F-Value:", f_value)

    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return y_proba, y_test, p_values, r_squared, f_value









############################################################################################################################################


def get_the_probabilities_with_logistic_regressionn_new(df, n1, n2, n3, n4, n5, n6, n7, n8, printt, use_df1="yes", use_df2="yes", use_df3="yes", use_df4="no"):
    # Create all dataframes as before
    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))
    
    # Create train and test dataframes based on the parameters
    train_dfs = []
    test_dfs = []
    
    if use_df1 == "yes":
        train_dfs.append(df1)
    else:
        test_dfs.append(df1)
        
    if use_df2 == "yes":
        train_dfs.append(df2)
    else:
        test_dfs.append(df2)
    
    if use_df3 == "yes":
        train_dfs.append(df3)
    else:
        test_dfs.append(df3)
    
    if use_df4 == "yes":
        train_dfs.append(df4)
    else:
        test_dfs.append(df4)
    
    # Concatenate dataframes for training and testing
    df_log_reg_train = pd.concat(train_dfs, ignore_index=True).copy() if train_dfs else pd.DataFrame()
    df_log_reg_test = pd.concat(test_dfs, ignore_index=True).copy() if test_dfs else pd.DataFrame()
    
    # Continue with the existing code
    y_train = df_log_reg_train["condition"]
    X_train = df_log_reg_train.drop(["condition", "timestamp"], axis=1)
    
    y_test = df_log_reg_test["condition"]
    X_test = df_log_reg_test.drop(["condition", "timestamp"], axis=1)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Calculate p-values
    from scipy import stats
    
    # Get the standard errors
    predictions = model.predict(X_train)
    residuals = y_train - predictions
    
    # Calculate variance residual and standard error
    mse = np.sum(residuals**2) / (len(y_train) - X_train.shape[1] - 1)

    # DENEME SONRA SİL  SATIR
    print(np.linalg.matrix_rank(np.dot(X_train.T, X_train)))
    print(np.dot(X_train.T, X_train).shape)
    corr_matrix = pd.DataFrame(X_train).corr()
    print(corr_matrix)
    #

    var_coef = mse * np.linalg.pinv(np.dot(X_train.T, X_train)).diagonal()
    se_coef = np.sqrt(var_coef)
    
    # Calculate z-scores and p-values
    z_scores = model.coef_[0] / se_coef
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_scores]
    
    # Calculate R-squared
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_train)
    r_squared = r2_score(y_train, y_pred)
    
    # Calculate F-value
    from sklearn.metrics import mean_squared_error
    mse_model = mean_squared_error(y_train, y_pred)
    mse_baseline = np.var(y_train)
    f_value = (mse_baseline - mse_model) / mse_model * (len(y_train) - X_train.shape[1] - 1) / X_train.shape[1]
    
    feature_names = X_train.columns
    coef_df = pd.DataFrame({
        'Coefficient': model.coef_[0],
        'p_value': p_values
    }, index=feature_names)
    
    coef_df_sorted = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    if printt == "yes":
        print(coef_df_sorted)
        print("Intercept:", model.intercept_[0])
        print("R-squareddd:", r_squared)
        print("F-Value:", f_value)

    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return y_proba, y_test, p_values, r_squared, f_value



def group_rows_by_condition_sliding(df, group_size=400, slide_amount=100):
    total_rows = len(df)
    
    # Calculate number of sliding windows
    num_groups = max(1, int(np.ceil((total_rows - group_size) / slide_amount)) + 1) if total_rows >= group_size else 1
    
    result_data = []
    
    for i in range(num_groups):
        # Calculate sliding window indices
        start_idx = i * slide_amount
        end_idx = min(start_idx + group_size, total_rows)
        
        # Stop if we've reached the end
        if start_idx >= total_rows:
            break
        
        current_group = df.iloc[start_idx:end_idx]
        
        # Gruptaki condition değerlerini kontrol ediyoruz
        if 1 in current_group['condition'].values:
            group_condition = 1
        else:
            group_condition = 0
        
        # Grubun ortalama proba değerini hesaplıyoruz
        group_proba = current_group['proba'].mean()
        
        # Yeni satırı sonuç listesine ekliyoruz
        result_data.append({
            'group_id': i,
            'start_row': start_idx,
            'end_row': end_idx - 1,
            'row_count': end_idx - start_idx,
            'proba': group_proba,
            'condition': group_condition
        })
    
    # Sonuç listesini DataFrame'e dönüştürüyoruz
    result_df = pd.DataFrame(result_data)
    
    return result_df





def get_the_probabilities_with_random_forest_new(number_of_estimators, max_features, df, n1, n2, n3, n4, n5, n6, n7, n8, printt, use_df1="yes", use_df2="yes", use_df3="yes", use_df4="no"):
    # Create all dataframes as before
    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))
    
    # Create train and test dataframes based on the parameters
    train_dfs = []
    test_dfs = []
    
    if use_df1 == "yes":
        train_dfs.append(df1)
    else:
        test_dfs.append(df1)
        
    if use_df2 == "yes":
        train_dfs.append(df2)
    else:
        test_dfs.append(df2)
    
    if use_df3 == "yes":
        train_dfs.append(df3)
    else:
        test_dfs.append(df3)
    
    if use_df4 == "yes":
        train_dfs.append(df4)
    else:
        test_dfs.append(df4)
    
    # Concatenate dataframes for training and testing
    df_rf_train = pd.concat(train_dfs, ignore_index=True).copy() if train_dfs else pd.DataFrame()
    df_rf_test = pd.concat(test_dfs, ignore_index=True).copy() if test_dfs else pd.DataFrame()
    
    # Continue with the existing code
    y_train = df_rf_train["condition"]
    X_train = df_rf_train.drop(["condition", "timestamp"], axis=1)
    
    y_test = df_rf_test["condition"]
    X_test = df_rf_test.drop(["condition", "timestamp"], axis=1)
    
    from sklearn.ensemble import RandomForestClassifier
    


    model = RandomForestClassifier(
    n_estimators= number_of_estimators,
    max_features= max_features,     # tüm feature'lara bak
    bootstrap=False,       # örnekleme yok
    random_state=42)

    model.fit(X_train, y_train)





    model.fit(X_train, y_train)
    
    # Feature importances
    feature_importances = model.feature_importances_
    feature_names = X_train.columns
    
    importance_df = pd.DataFrame({
        'Importance': feature_importances
    }, index=feature_names)
    
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)
    
    # Calculate R-squared (not perfect for classification but can be used on predict_proba output)
    from sklearn.metrics import r2_score
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    r_squared = r2_score(y_train, y_pred_proba_train)
    
    # Calculate F-value-like metric (optional; Random Forests don't naturally have it, but let's keep the spirit)
    from sklearn.metrics import mean_squared_error
    mse_model = mean_squared_error(y_train, y_pred_proba_train)
    mse_baseline = np.var(y_train)
    f_value_like = (mse_baseline - mse_model) / mse_model * (len(y_train) - X_train.shape[1] - 1) / X_train.shape[1]
    
    if printt == "yes":
        print(importance_df_sorted)
        print("R-squareddd:", r_squared)
        print("F-Value Like:", f_value_like)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return y_proba, y_test, feature_importances, r_squared, f_value_like

#İŞE YARAMAZSA AŞAĞIYI SİL

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict

def get_the_probabilities_with_rf_logreg_leaf(df, n1, n2, n3, n4, n5, n6, n7, n8,
                                              printt="no", use_df1="yes", use_df2="yes", use_df3="yes", use_df4="no"):
    # Timestamp'lara göre filtreleme yapılmaktadır.
    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))
    
    # Eğitim ve test setleri belirlenmektedir.
    train_dfs, test_dfs = [], []
    if use_df1 == "yes": train_dfs.append(df1)
    else: test_dfs.append(df1)
    if use_df2 == "yes": train_dfs.append(df2)
    else: test_dfs.append(df2)
    if use_df3 == "yes": train_dfs.append(df3)
    else: test_dfs.append(df3)
    if use_df4 == "yes": train_dfs.append(df4)
    else: test_dfs.append(df4)
    
    df_rf_train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    df_rf_test = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    y_train = df_rf_train["condition"].values
    X_train = df_rf_train.drop(["condition", "timestamp"], axis=1).values
    y_test = df_rf_test["condition"].values
    X_test = df_rf_test.drop(["condition", "timestamp"], axis=1).values
    feature_names = df_rf_train.drop(["condition", "timestamp"], axis=1).columns

    # Random Forest modeli segmentasyon için eğitilmektedir.
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # Eğitim verisindeki yaprak indeksleri alınmaktadır.
    train_leaf_indices = rf.apply(X_train)  # shape: (n_samples, n_trees)
    test_leaf_indices = rf.apply(X_test)    # shape: (n_samples, n_trees)

    # Her yaprak için logistic regression modelleri eğitilmektedir.
    leaf_models = defaultdict(dict)
    for tree_idx in range(train_leaf_indices.shape[1]):
        for leaf_idx in np.unique(train_leaf_indices[:, tree_idx]):
            mask = train_leaf_indices[:, tree_idx] == leaf_idx
            X_leaf = X_train[mask]
            y_leaf = y_train[mask]
            
            if len(np.unique(y_leaf)) < 2:
                continue  # Logistic regression yapılamaz
            
            model = LogisticRegression()
            model.fit(X_leaf, y_leaf)
            leaf_models[tree_idx][leaf_idx] = model

    # Test seti için olasılık tahmini yapılmaktadır.
    y_proba = np.zeros(X_test.shape[0])
    for tree_idx in range(test_leaf_indices.shape[1]):
        for leaf_idx in np.unique(test_leaf_indices[:, tree_idx]):
            mask = test_leaf_indices[:, tree_idx] == leaf_idx
            X_leaf_test = X_test[mask]
            model = leaf_models[tree_idx].get(leaf_idx)
            if model:
                y_proba[mask] += model.predict_proba(X_leaf_test)[:, 1]
    
    y_proba /= test_leaf_indices.shape[1]  # Tüm ağaçların ortalaması alınmaktadır.

    # Train seti üzerinden feature importance alınmaktadır (isteğe bağlı).
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({'Importance': importances}, index=feature_names).sort_values(by='Importance', ascending=False)

    # Performans metrikleri hesaplanmaktadır.
    y_pred_proba_train = rf.predict_proba(X_train)[:, 1]
    r_squared = r2_score(y_train, y_pred_proba_train)
    mse_model = mean_squared_error(y_train, y_pred_proba_train)
    mse_baseline = np.var(y_train)
    f_value_like = (mse_baseline - mse_model) / mse_model * (len(y_train) - X_train.shape[1] - 1) / X_train.shape[1]
    
    if printt == "yes":
        print(importance_df)
        print("R-squared:", r_squared)
        print("F-Value Like:", f_value_like)

    return y_proba, y_test, importances, r_squared, f_value_like




# This function creates train and test datasets based on the provided time intervals, just like before.
# However, instead of fitting a RandomForestClassifier, it fits a single DecisionTreeClassifier
# so that the inner workings of the model can be inspected more clearly.
# The function also plots the decision tree to provide a visual understanding of the splits.
# Feature importance and R-squared metrics are calculated and printed as before.

def get_the_probabilities_with_single_tree(df, n1, n2, n3, n4, n5, n6, n7, n8, printt, use_df1="yes", use_df2="yes", use_df3="yes", use_df4="no"):
    
    df1 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f1_start, "backward", n1), adjust_datetime(f1_finish, "forward", n2))
    df2 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f2_start, "backward", n3), adjust_datetime(f2_finish, "forward", n4))
    df3 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f3_start, "backward", n5), adjust_datetime(f3_finish, "forward", n6))
    df4 = filter_rows_between_the_given_timestamps(df, adjust_datetime(f4_start, "backward", n7), adjust_datetime(f4_finish, "forward", n8))

    train_dfs = []
    test_dfs = []

    if use_df1 == "yes":
        train_dfs.append(df1)
    else:
        test_dfs.append(df1)

    if use_df2 == "yes":
        train_dfs.append(df2)
    else:
        test_dfs.append(df2)

    if use_df3 == "yes":
        train_dfs.append(df3)
    else:
        test_dfs.append(df3)

    if use_df4 == "yes":
        train_dfs.append(df4)
    else:
        test_dfs.append(df4)

    df_train = pd.concat(train_dfs, ignore_index=True).copy() if train_dfs else pd.DataFrame()
    df_test = pd.concat(test_dfs, ignore_index=True).copy() if test_dfs else pd.DataFrame()

    y_train = df_train["condition"]
    X_train = df_train.drop(["condition", "timestamp"], axis=1)

    y_test = df_test["condition"]
    X_test = df_test.drop(["condition", "timestamp"], axis=1)

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    feature_names = X_train.columns

    importance_df = pd.DataFrame({
        'Importance': feature_importances
    }, index=feature_names).sort_values(by='Importance', ascending=False)

    from sklearn.metrics import r2_score
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    r_squared = r2_score(y_train, y_pred_proba_train)

    from sklearn.metrics import mean_squared_error
    mse_model = mean_squared_error(y_train, y_pred_proba_train)
    mse_baseline = np.var(y_train)
    f_value_like = (mse_baseline - mse_model) / mse_model * (len(y_train) - X_train.shape[1] - 1) / X_train.shape[1]

    if printt == "yes":
        print(importance_df)
        print("R-squared:", r_squared)
        print("F-Value Like:", f_value_like)

    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["No", "Yes"], filled=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.show()

    y_proba = model.predict_proba(X_test)[:, 1]

    return y_proba, y_test, feature_importances, r_squared, f_value_like
