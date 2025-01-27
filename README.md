# Seoul Bike Sharing Demand Prediction
## Overview
Code and data used to model rental bike demand in Seoul, South Korea. Both daily and hourly demand are modelled, based largely on environmental data (temperture, humidity etc.). 

Data was obtained from [here](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), with a journal article discussing the dataset and several modelling techniques found [here](https://www.sciencedirect.com/science/article/abs/pii/S0140366419318997).

## Project Status
Still in development. So far only a simple polynomial regression model has been implemented. In the future, I plan to test various more expressive models (e.g. SVM, Decision Trees etc.).

## Requirements 
The following packages are needed:
- `sklearn`
- `pandas` 
