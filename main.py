from src import model_linear, preprocessing
import pandas as pd

data_file = 'SeoulBikeData.csv'
df = pd.read_csv(data_file, encoding='ISO-8859-1')

preprocessing = preprocessing.Preprocessing(df)
df_clean = preprocessing.text_to_num()

# remove data for days when its not possible to rent a bike
df_clean = df_clean.loc[df_clean['Functioning Day'] == 1.0]

linear_model = model_linear.ModelLinear()

# separate out features and target
features = df_clean.drop(columns=["Rented Bike Count", "Functioning Day"] )
target = df_clean["Rented Bike Count"]

# split data into train/test/validation
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3,
                                            random_state = 1)


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
                    feature_df = x_train,
                    target = y_train)
