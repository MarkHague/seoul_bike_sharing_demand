from pyexpat import features

from requests.packages import target
from src import model_linear, preprocessing
import pandas as pd

data_file = 'SeoulBikeData.csv'
df = pd.read_csv(data_file, encoding='ISO-8859-1')

preprocessing = preprocessing.Preprocessing(df)
df_clean = preprocessing.text_to_num()

linear_model = model_linear.ModelLinear()

# train the model to predict hourly data
print("Hourly Data:")
trained_model_hr_data = linear_model.train_model(
                    feature_df = df_clean.drop(columns=["Rented Bike Count", "Functioning Day","Date"]),
                    target = df_clean["Rented Bike Count"])

# train on daily data
df_clean_daily = preprocessing.downsample()
# define a new model with less polynomial terms, since we have reduced the complexity
linear_model_daily = model_linear.ModelLinear(n_poly = 3)
print("Daily Data:")
trained_model_daily_data = linear_model_daily.train_model(
                    feature_df = df_clean_daily.drop(columns=["Rented Bike Count", "Functioning Day"]),
                    target = df_clean_daily["Rented Bike Count"])