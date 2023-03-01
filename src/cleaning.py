import pandas as pd
import yaml
from src.utils.cleaning_utils import (Fill_missing_Val_Columns,
                                      Drop_Missing_Val_Columns)
from ml_service.utils.env_variables import Env

e = Env()

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

traindf = pd.read_parquet(e.train_file_path, engine='pyarrow')
print(traindf.head())

traindf = Fill_missing_Val_Columns(traindf,
                                   config['MISSING_VAL_COLUMNS_TO_FILL'])
traindf = Drop_Missing_Val_Columns(traindf, config['MISSING_VAL_COLUMNS'])
traindf_cleaned = traindf
traindf_cleaned.to_parquet(e.train_cleaned_file_path, index=False)
