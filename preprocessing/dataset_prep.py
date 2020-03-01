import pandas as pd
import glob
import os
from functools import reduce


class DatasetPrep:
    def __init__(self, files_dir, out_path, minimum_data_length=10,
                 minimum_num_gps_prescribed_drug=10000, minimum_date_count=57):
        self.files_dir = files_dir
        self.out_path = out_path
        self.unnamed = 'Unnamed: 0'
        self.gp_code_column = 'GPs'
        # minimum size number of file to use as threshold to spot empty datasets
        self.minimum_data_length = minimum_data_length  
        # minimum number of gps prescribing the drug
        self.minimum_num_gps_prescribed_drug = minimum_num_gps_prescribed_drug
        # each GP has to have a minimum of ocurrences
        self.minimum_date_count = minimum_date_count
        # getting all datasets filenames
        self.csv_files = self.get_csv_filenames()
        # building dataframe
        self.df = self.build_dataframe()
        # saving dataset to a file
        self.save_to_file()

    def get_csv_filenames(self):
        csv_files = glob.glob(os.path.join(self.files_dir, '*.csv'))
        # remove empty files
        csv_files = [csv_file for csv_file in csv_files
                     if os.stat(csv_file).st_size > self.minimum_data_length]
        return csv_files

    def wrangling_dataset(self, csv_file):
        # read dataset
        df = pd.read_csv(csv_file)
        # discard data that has not been used for minimum_number_gps or dementia id
        if df.shape[1] > self.minimum_num_gps_prescribed_drug or csv_file.split('/')[-1] == '0411.csv':
            df = df.set_index(self.unnamed)
            df = df.T
            df = df.reset_index(level=0)
            df = df.rename(columns={"index": self.gp_code_column})
            df.columns.names = ['']
            df = pd.melt(df, id_vars=self.gp_code_column,
                         var_name='date',
                         value_name='num_drugs_prescribed')
            df['drug_id'] = csv_file.split('/')[-1][:-4]
            df = self.recasting_columns(df)
            df = df.set_index(['GPs', 'date'])
            return df
        else:
            return pd.DataFrame(columns=None)

    def recasting_columns(self, df):
        # drug_ids as columns for num_drugs_prescribed
        df = df.pivot_table(index=['GPs', 'date'],
                            columns='drug_id',
                            values='num_drugs_prescribed').reset_index()
        df.columns.names = ['']
        return df

    def build_dataframe(self):
        dfs = [self.wrangling_dataset(csv_file) for csv_file in self.csv_files]
        dfs = [x for x in dfs if not x.empty]
        # joining datasets by GPs and time
        df = reduce(self.join_dfs, dfs)
        df = df.reset_index()
        # Filtering minimum date number of drug prescription
        df = df.groupby('GPs').filter(lambda x: len(x) > self.minimum_date_count)
        df = df.sort_values(by=['GPs', 'date'])
        # Interpolating NaN values
        df = df.groupby('GPs').apply(lambda x: x.interpolate(method='linear',
                                                             limit_direction='both'))
        # Getting month from date
        df['date'] = [int(d.split('-')[1]) for d in df.date]
        df = df.rename(columns={'date': 'month'})
        
        # Deleting other appliances drugs
        df = df.drop(columns=['2101'])
        
        # Due dementia case (less than 10000 GPs)
        df = df.dropna(axis='index')

        return df

    def join_dfs(self, d1, d2):
        return d1.join(d2)

    def save_to_file(self):
        self.df.to_csv(self.out_path)


files_dir = '../data/raw/NHS_challenge_data'
out_path = '../data/processed/processed_cleaned_df.csv'
DatasetPrep(files_dir, out_path)

