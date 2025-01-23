import pandas as pd

class Preprocessing:

    def __init__(self, df):
        self.df = df
        self.start_date = '2017-12-01'
        self.end_date = '2018-11-30 23:00:00'
        self.input_freq = 'h'

    def text_to_num(self):
        """Convert text features to numbers that can be used for training"""
        # convert "Holiday" into binary: 0 = "No Holiday", 1 = "Holiday"
        self.df.loc[self.df.Holiday == "No Holiday", "Holiday"] = 0
        self.df.loc[self.df.Holiday == "Holiday", "Holiday"] = 1
        # convert "Functioning Day" into binary: 1 = "Yes", 0 = "No"
        self.df.loc[self.df["Functioning Day"] == "Yes", "Functioning Day"] = 1
        self.df.loc[self.df["Functioning Day"] == "No", "Functioning Day"] = 0
        # map seasons onto numeric values
        seasons_dict = {
            "Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3
        }
        self.df.replace({"Seasons": seasons_dict}, inplace=True)

        return self.df

    def downsample(self, freq='D'):
        """Aggregate hourly data to courser resolution"""
        # downsample data to daily
        dates = pd.date_range(self.start_date, self.end_date, freq=self.input_freq)
        self.df.set_index(dates, inplace=True)
        self.df = self.df.drop(columns=["Date", "Hour"]).resample(freq).mean()
        # for the target variable, we probably want the total bikes rented, not the mean
        self.df["Rented Bike Count"] = self.df.resample('D').sum()["Rented Bike Count"]
        # remove data where functioning data is "No" (now 0)
        self.df.drop(self.df[self.df["Functioning Day"] != 1.0].index, inplace=True)

        return self.df

