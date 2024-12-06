## My Project

My project focuses on analyzing and predicting greenhouse gas (GHG) emissions using a combination of exploratory data analysis (EDA), machine learning (ML), and neural networks. This work integrates environmental and economic datasets to gain insights into regional emissions, identify key influencing factors, and predict future emissions trends.


***

## Introduction 

Greenhouse gas emissions have become one of the most pressing issues of our time, contributing significantly to global warming and climate change. Understanding and predicting these emissions are critical for developing effective policies and strategies to combat climate change.

The problem is that emissions data are often complex, involving numerous regions, sectors, and influencing economic factors. Traditional methods struggle to capture these relationships effectively.

To address this issue, we use machine learning to analyze and predict emissions. The dataset includes detailed regional and sectoral data, making supervised learning the most suitable approach for this task. By leveraging advanced ML models, we aim to:
1. Identify key factors driving emissions.
2. Predict emissions trends and hotspots for future periods.

We analyzed the data, trained predictive models, and concluded that ML can effectively identify emissions drivers and provide accurate forecasts. This work highlights the potential for machine learning in climate action planning.

## Data

The dataset used in this project combines environmental and economic data, merging information from two sources:
1. **ExioML_factor_accounting_IxI.csv**
2. **ExioML_factor_accounting_PxP.csv**

These datasets provide:
- Regional and sectoral identifiers.
- Economic indicators such as value-added contributions and energy usage.
- Historical greenhouse gas emissions data.

### Data Preprocessing
- Missing values were imputed using mean or most frequent values, depending on the feature type.
- Categorical features (`region` and `sector`) were label-encoded.
- Numerical features (e.g., `GHG emissions`, `Value Added`) were normalized to ensure compatibility with machine learning models.

Below is an example visualization from the dataset:

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Emissions data distribution, visualized for exploratory analysis.*

***

## Modelling

### Machine Learning Approach
We chose supervised learning methods to predict emissions. Key approaches included:
1. **Random Forest Regressor**: For baseline predictions and feature importance analysis.
2. **LightGBM**: For efficient, high-performance modeling with categorical and numerical data.
3. **LSTM Neural Networks**: To predict future emissions using temporal trends.

The models optimize for mean squared error (MSE), providing a quantitative measure of performance. Below is an example code snippet used for the Random Forest model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

##Model Development
Data was split into training and testing sets to evaluate performance.
Hyperparameters were tuned to optimize model performance.
Feature importance analysis revealed that regional identifiers and economic indicators were the most significant drivers of emissions.

## Results

Figure X shows the GHG emissions by region, as predicted by our models.
Figure X: Predicted GHG emissions by region in 2022.
Key results include:
High-emission regions were accurately identified.
Temporal trends showed a decrease in emissions for certain regions while others exhibited growth.
Models achieved competitive MSE scores, with LightGBM outperforming other approaches.

## Discussion

From Figure X, we can observe that certain regions contribute disproportionately to emissions. The feature importance analysis highlighted that economic activity, energy usage, and regional identifiers significantly impact emissions.
The results demonstrate the utility of machine learning in identifying patterns and predicting future emissions. However, limitations include:
Dependence on data quality: Missing or inaccurate data can affect model reliability.
Temporal generalization: LSTM predictions are sensitive to historical trends and require careful tuning.

## Conclusion

From this work, the following conclusions can be made:
Machine learning effectively identifies key drivers of GHG emissions.
Predictive models like LightGBM and LSTM can forecast emissions trends accurately.
Visualization techniques provide actionable insights for policymakers.


Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

