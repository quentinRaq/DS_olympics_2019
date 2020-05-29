import re
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.target_encoder import TargetEncoder
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def optimize_csv(df, dates=None, verbose=1):
    if dates is None:
        dates = []
    if verbose > 0:
        print('------------------General info---------------------' + '\n')
    for dtype in ['float','int','object']:
        selected_dtype = df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).sum()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        if verbose > 0:
            print("Sum of memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
        
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric,downcast='unsigned')

    if verbose > 1:
        print('\n' + '-----------------------INT-------------------------')
        print('---------------------------------------------------' + '\n')
        print('Memory usage of int (before optimization): {}'.format(mem_usage(df_int)))
        print('Memory usage of int (after optimization): {}'.format(mem_usage(converted_int)))


    compare_ints = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)
    compare_ints.columns = ['before','after']
    if verbose > 1:
        print('\n' + '-----------------Transformation--------------------')
        print(compare_ints.apply(pd.Series.value_counts))

    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric,downcast='float')

    if verbose > 1:
        print('\n' + '---------------------FLOAT-------------------------')
        print('---------------------------------------------------' + '\n')
        print('Memory usage of float (before optimization): {}'.format(mem_usage(df_float)))
        print('Memory usage of float (after optimization): {}'.format(mem_usage(converted_float)))
        print('---------------------------------------------------')

    compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes],axis=1)
    compare_floats.columns = ['before','after']
    if verbose > 1:
        print('\n' + '-----------------Transformation--------------------')
        print(compare_floats.apply(pd.Series.value_counts))
          
          
    df_obj = df.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in df_obj.columns:
        num_unique_values = len(df_obj[col].unique())
        num_total_values = len(df_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:,col] = df_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = df_obj[col]
          
    if verbose > 1:
        print('\n' + '---------------------OBJECT------------------------')
        print('---------------------------------------------------' + '\n')
        print('Memory usage of object (before optimization): {}'.format(mem_usage(df_obj)))
        print('Memory usage of object (after optimization): {}'.format(mem_usage(converted_obj)))
        print('---------------------------------------------------')

    compare_obj = pd.concat([df_obj.dtypes,converted_obj.dtypes],axis=1)
    compare_obj.columns = ['before','after']
    if verbose > 1:
        print('\n' + '-----------------Transformation--------------------')
        print(compare_obj.apply(pd.Series.value_counts))
          
    optimized_df = df.copy()

    optimized_df[converted_int.columns] = converted_int
    optimized_df[converted_float.columns] = converted_float
    optimized_df[converted_obj.columns] = converted_obj

    if verbose > 0:
        print( '\n' + '---------------------GLOBAL------------------------')
        print('---------------------------------------------------' + '\n')
        print('Memory usage of dataset (before optimization): {}'.format(mem_usage(df)))
        print('Memory usage of dataset (after optimization): {}'.format(mem_usage(optimized_df)))
          
    dtypes = optimized_df.drop(dates,axis=1).dtypes
    del optimized_df
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]
    column_types = dict(zip(dtypes_col, dtypes_type))
    
    return column_types


# Parallel computation
def do_parallel(func, params, max_workers=7):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        new_features = executor.map(func, params)
        del params
    return new_features


# tfidf
def get_tfidf_vectorizer(params):
    col = params[0]
    tokenizer = params[1]
    language = params[2]
    max_features = params[3]
    stop_words = list(stopwords.words(language))
    vectorizer = TfidfVectorizer(analyzer='word', 
                                 ngram_range=(1,3), 
                                 stop_words = stop_words, 
                                 lowercase=True, 
                                 max_features=max_features, 
                                 binary=True, 
                                 norm=None, 
                                 use_idf=False,
                                 tokenizer=tokenizer)
    tfidf = vectorizer.fit_transform(col)
    tfidf_cols = vectorizer.get_feature_names()
    clean_text = lambda x: re.sub(r'[^\w\s]','', str(x).lower()).replace(' ', '_')
    tfidf_cols = list(map(clean_text, tfidf_cols))
    res = pd.DataFrame(data=tfidf.todense(),
                       columns=['tfidf_' + col.name + '_' + str(i) for i in tfidf_cols],
                       index=col.index)
    print(f'Creation of tfidf features for {col.name}')
    return res

#custom tokenizer for tfifd
def custom_tokenizer(s):
    return s.split(',')

#extract text feature
def get_text_feature(params):
    col = params[0]
    language = params[1]
    stop_words = list(stopwords.words(language))
    #lowering and removing punctuation
    tmp = col.apply(lambda x: re.sub(r'[^\w\s]','', str(x).lower()))
    df = {}
    #numerical feature engineering
    #total length of sentence
    df['{}_length'.format(col.name)] = tmp.apply(lambda x: len(x))
    #get number of words
    df['{}_words'.format(col.name)] = tmp.apply(lambda x: len(x.split(' ')))
    df['{}_words_not_stopword'.format(col.name)] = tmp.apply(lambda x: len([t for t in x.split(' ') if t not in stop_words]))
    #get the average word length
    df['{}_avg_word_length'.format(col.name)] = tmp.apply(lambda x: np.mean([len(t) for t in x.split(' ') if t not in stop_words]) if len([len(t) for t in x.split(' ') if t not in stop_words]) > 0 else 0)
    #get the average word length
    df['{}_commas'.format(col.name)] = col.apply(lambda x: str(x).count(','))
    print(f'Creation of text features for {col.name}')
    return pd.DataFrame(df, index=col.index)


#distinct count agregate
def get_distinct_count(params):
    df = params[0]
    field = params[1]
    by_field = params[2]
    tmp = df.copy()
    if tmp[field].dtype.name == 'category':
        tmp[field] = tmp[field].cat.add_categories(['xxx'])
    tmp[field].fillna('xxx', inplace=True)
    tmp.drop_duplicates(inplace=True)
    tmp = tmp.groupby([by_field]).count()[[field]].reset_index()
    new_name = 'distinct_count_of_' + field + '_by_' + str(by_field)
    tmp.columns = [i for i in [by_field]]+[new_name]
    res = df[by_field].to_frame().merge(tmp, on=by_field, how='left')
    print(f'Creation of {new_name}')
    return pd.Series(res[new_name].values, index=df.index, name=new_name) 


#mode agregate
def get_mode(params):
    df = params[0]
    field = params[1]
    by_field = params[2]
    tmp = df.copy()
    if tmp[field].dtype.name == 'category':
        tmp[field] = tmp[field].cat.add_categories(['xxx'])
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp.groupby([by_field]).agg(lambda x:x.value_counts().index[0])[[field]].reset_index()
    new_name = 'mode_' + field + '_by_' + str(by_field)
    tmp.columns = [i for i in [by_field]]+[new_name]
    res = df[by_field].to_frame().merge(tmp, on=by_field, how='left')
    print(f'Creation of {new_name}')
    return pd.Series(res[new_name].values, index=df.index, name=new_name) 


def get_min(params):
    df = params[0]
    field = params[1]
    by_field = params[2]
    tmp = df.copy()
    if tmp[field].dtype.name == 'category':
        tmp[field] = tmp[field].cat.add_categories(['xxx'])
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp.groupby([by_field]).min()[[field]].reset_index()
    new_name = 'min_' + field + '_by_' + str(by_field)
    tmp.columns = [i for i in [by_field]]+[new_name]
    res = df[by_field].to_frame().merge(tmp, on=by_field, how='left')
    print(f'Creation of {new_name}')
    return pd.Series(res[new_name].values, index=df.index, name=new_name) 


#mode agregate
def get_max(params):
    df = params[0]
    field = params[1]
    by_field = params[2]
    tmp = df.copy()
    if tmp[field].dtype.name == 'category':
        tmp[field] = tmp[field].cat.add_categories(['xxx'])
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp.groupby([by_field]).max()[[field]].reset_index()
    new_name = 'max_' + field + '_by_' + str(by_field)
    tmp.columns = [i for i in [by_field]]+[new_name]
    res = df[by_field].to_frame().merge(tmp, on=by_field, how='left')
    print(f'Creation of {new_name}')
    return pd.Series(res[new_name].values, index=df.index, name=new_name) 


#date feature creation
def get_date_features(params):
    col = params[0]
    hour_bool = params[1]
    days_bool = params[2]
    month_bool = params[3]
    year_bool = params[4]
    new_cols = {}
    if hour_bool:
        col_hour = col.dt.hour.astype('str')
        new_cols[col.name + '_hour'] = col_hour
    if days_bool:
        col_days = col.dt.weekday_name
        new_cols[col.name + '_day'] = col_days
    if month_bool:
        col_month = col.dt.month.astype('str')
        new_cols[col.name + '_month'] = col_month
    if year_bool:
        col_year = col.dt.year.astype('str')
        new_cols[col.name + '_year'] = col_year
    print(f'Creation of date features for {col.name}')
    return pd.DataFrame(new_cols, index=col.index)


def standard_scaler(params):
    train = params[0]
    test = params[1]
    ss = StandardScaler()
    train = ss.fit_transform(train.reshape(-1, 1))
    test = ss.transform(test.reshape(-1, 1))
    return train.flatten(), test.flatten()


def modality_grouper(params):
    train = params[0].astype('str')
    test = params[1].astype('str')
    threshold = params[2]
    mg = Modality_grouper(thres=threshold)
    mg.fit(train)
    train = mg.transform(train)
    test = mg.transform(test)
    return train.flatten(), test.flatten()


def target_encoder(params):
    train = params[0].astype('str')
    test = params[1].astype('str')
    target = params[2]
    te = TargetEncoder(return_df=False)
    train = te.fit_transform(train.reshape(-1, 1), target.reshape(-1, 1))
    test = te.transform(test.reshape(-1, 1))
    return train.flatten(), test.flatten()


def ordinal_encoder(params):
    train = params[0].astype('str')
    test = params[1].astype('str')
    oe = OrdinalEncoder()
    train = oe.fit_transform(train.reshape(-1, 1))
    test = oe.transform(test.reshape(-1, 1))
    return train.flatten(), test.flatten()


def one_hot_encoder(params):
    train = params[0].astype('str')
    test = params[1].astype('str')
    col_name = params[2]
    oh = OneHotEncoder(sparse=False)
    new_train = oh.fit_transform(train.values.reshape(-1, 1))
    new_test = oh.transform(test.values.reshape(-1, 1))
    clean_text = lambda x: re.sub(r'[^\w\s]','', str(x).lower()).replace(' ', '_')
    new_names = list(map(clean_text, list(oh.categories_[0])))
    new_col_names = [col_name + '_' + new_name[:20] for new_name in new_names]
    return (pd.DataFrame(new_train, index=train.index, columns=new_col_names),
            pd.DataFrame(new_test, index=test.index, columns=new_col_names))


class Modality_grouper(BaseEstimator, TransformerMixin):
    """Combine levels of low frequency modalities.

    Parameters
    ----------
    max_cat : integer (default=50)
        Maximum modality number to keep. Low frequency modalities will be
        considered as 'other'.
    """

    def __init__(self, thres=20):
        self.thres = thres

    def fit(self, X, y=None):
        """Fit the Modality_grouper on X.

        Parameters
        ----------
        X : {array-like, list, Series}, shape (n_samples)
            Input data, where ``n_samples`` is the number of samples.
        Returns
        -------
        self : Custom_fillna
            Returns self.

        """
        series = pd.Series(X)
        categories = series.value_counts(dropna=False).reset_index()
        categories.columns = ['id', 'var']
        categories.sort_values(by=['var', 'id'], ascending=False, inplace=True)
        self.list_cat = list(categories.loc[categories['var'] > self.thres, 'id'].values)

        return self

    def transform(self, X):
        """Replace values not in self.list_cat by 'OtherValue'.

        Parameters
        ----------
        X : {array-like, list, Series}, shape (n_samples)
            Input data, where ``n_samples`` is the number of samples.

        """
        try:
            assert(hasattr(self, 'list_cat'))
        except AssertionError as err:
            raise type(err)('Modality_grouper is not fitted')
        res = [str(x) + '_' if x in self.list_cat else 'OtherValue' for x in X]

        return np.array(pd.Series(res)).reshape(-1, 1)
