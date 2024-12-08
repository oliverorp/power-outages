# Utilities Industry in the US: Risk Factors and Predictive Modeling

A Data Analysis and ML HW for EECS 398 at U-M

<small>Oliver Pourmussa | </small>
<small> oliverpo@umich.edu </small>

## <span style="color:#FF9999;"> Introduction<span>

The data set I am working with consists of power outages in the US from 2000 - 2016. There is much to be deduced from this data, but I am particularly curious as to the risk climate of the utilities industry.

What geographic regions are most prone to severe power outages, and more generally, what are some things we can discover about the challenges faced in the utilities industry?

In the interest of deploying my machine learning tool kit, I will also build a multiclass classifier that determines the cause of a given outage.

It is this kind of inquiry that finds its place both in the interest of an investor and a utilities company. Investors are concerned about risk, so the exploratory data analysis section is fitting. Utility companies need to be prepared with how to deploy resources and address outages, so the final machine learning model may have some applications to that demographic as well.

### Dataset

Number of Rows: 1534

| **Column Name**      | **Description**                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `YEAR`               | The year in which the data was recorded.                                                                                       |
| `POSTAL.CODE`        | The state abbreviation.                                                                                                        |
| `CLIMATE.REGION`     | The climate region to which the location belongs.                                                                              |
| `ANOMALY.LEVEL`      | Level of climate anomaly oscillation.                                                                                          |
| `CAUSE.CATEGORY`     | The category of the cause for the outage (e.g., severe weather).                                                               |
| `OUTAGE.DURATION`    | The duration of the outage (minutes)                                                                                           |
| `CUSTOMERS.AFFECTED` | The number of customers affected by the outage.                                                                                |
| `TOTAL.CUSTOMERS`    | The total number of customers in the area or region.                                                                           |
| `POPULATION`         | The population of the area or region.                                                                                          |
| `POPPCT_URBAN`       | Percentage of the population living in urban areas.                                                                            |
| `PCT_WATER_TOT`      | The percentage of water area in a state compared to the total water area in the continental U.S (e.g., lakes, rivers).         |
| `SEVERE`             | Whether the outage is classified as severe or not (exceeding 1 day in duration). This column was added using a transformation. |

<br>
<br>
## <span style="color:#FF9999;"> Data Cleaning and Exploratory Data Analysis<span>

### Data Cleaning

Data was cleaned and imputed using a combination of strategies. The cleaning consisted of a few tasks: make the YEAR column a datetime object, make the climate regions for AK and HI the same as their state postal code since they were missing, and add a new column, SEVERE, as a boolean value indicating whether or not a power outage exceeded 1 day in duration.

The cleaned dataframe is shown below:

<div style="overflow-x: auto; max-width: 100%; padding: 10px;">
  <table>
    <thead>
      <tr>
        <th>YEAR</th>
        <th>POSTAL.CODE</th>
        <th>CLIMATE.REGION</th>
        <th>ANOMALY.LEVEL</th>
        <th>CAUSE.CATEGORY</th>
        <th>OUTAGE.DURATION</th>
        <th>CUSTOMERS.AFFECTED</th>
        <th>TOTAL.CUSTOMERS</th>
        <th>POPULATION</th>
        <th>POPPCT_URBAN</th>
        <th>PCT_WATER_TOT</th>
        <th>SEVERE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>2011</td>
        <td>MN</td>
        <td>East North Central</td>
        <td>-0.3</td>
        <td>severe weather</td>
        <td>3060</td>
        <td>70000</td>
        <td>2.5957e+06</td>
        <td>5.34812e+06</td>
        <td>73.27</td>
        <td>8.40733</td>
        <td>True</td>
      </tr>
      <tr>
        <td>2014</td>
        <td>MN</td>
        <td>East North Central</td>
        <td>-0.1</td>
        <td>intentional attack</td>
        <td>1</td>
        <td>0</td>
        <td>2.64074e+06</td>
        <td>5.45712e+06</td>
        <td>73.27</td>
        <td>8.40733</td>
        <td>False</td>
      </tr>
      <tr>
        <td>2010</td>
        <td>MN</td>
        <td>East North Central</td>
        <td>-1.5</td>
        <td>severe weather</td>
        <td>3000</td>
        <td>70000</td>
        <td>2.5869e+06</td>
        <td>5.3109e+06</td>
        <td>73.27</td>
        <td>8.40733</td>
        <td>True</td>
      </tr>
      <tr>
        <td>2012</td>
        <td>MN</td>
        <td>East North Central</td>
        <td>-0.1</td>
        <td>severe weather</td>
        <td>2550</td>
        <td>68200</td>
        <td>2.60681e+06</td>
        <td>5.38044e+06</td>
        <td>73.27</td>
        <td>8.40733</td>
        <td>True</td>
      </tr>
      <tr>
        <td>2015</td>
        <td>MN</td>
        <td>East North Central</td>
        <td>1.2</td>
        <td>severe weather</td>
        <td>1740</td>
        <td>250000</td>
        <td>2.67353e+06</td>
        <td>5.48959e+06</td>
        <td>73.27</td>
        <td>8.40733</td>
        <td>True</td>
      </tr>
    </tbody>
  </table>
</div>

### Imputation

In the dataframe above, several imputation strategies were employed. OUTAGE.DURATION had 58 NaN values, CUSTOMERS.AFFECTED had 443 NaNs, and ANOMALY.LEVEL had 9 NaN values. OUTAGE.DURATION was imputed using conditional median imputation based on outage cause since this distribution of duration is skewed right. ANOMALY.LEVEL was imputed using the mean, given its symmetric distribution. CUSTOMERS.AFFECTED was imputed using conditional median imputation, but was not utilized, with the exception of one visualization in the bivariate analysis section.

<iframe
  src="assets/anomaly_pre.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

<iframe
  src="assets/anomaly_post.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

The distribution of ANOMALY.LEVEL was left unchanged. This is because the distribution was already approximately normal.

<iframe
  src="assets/outage_duration_pre.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>
<iframe
  src="assets/outage_duration_post.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

The distribution of OUTAGE.DURATION accumulated more mass around its center point.

### Univariate Analysis

A simple bar chart displays the number of power outages associated with each cause category for the time period 2000 - 2016. It appears that severe weather, intentional attack, and system operability disruption are among the most frequent causes of power outages. This chart appeals to those interested in risk factors associated with the utilities industry and what remains to be the most significant hurdles in that space.

<iframe
  src="assets/univariate_1.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

## Bivariate Analysis

Let's continue our exploration of the risk factors in the utilities industry.

Specifically, let's explore the following

(1) Is the utilities industry adapting to its challenges? Let's look at customers affected over time to see if we can find a trend in outage mitigation.

(2) Distributions of outage duration by cause category in order to get a good understanding of what kinds of outages are the most severe.

<iframe
  src="assets/time_series.html"
  width="500"
  height="400"
  frameborder="0"
></iframe>

<iframe
  src="assets/box_plot.html"
  width="600"
  height="400"
  frameborder="0"
></iframe>

The time series data suggests a negative correlation between customers affected and time. Perhaps utilities companies are adapting to their challenges and becoming more efficient with outage mitigation.

The second bivariate display shows that severe weather, fuel supply emergency, and public appeal have the largest median response times. Markets especially prone to these risk factors may be of interest to someone seeking a better understanding of the risks associated with the utility space.

## Interesting Aggregates

Finally, on the topic of risks in the utilities space, we can group our data to see what regions are associated with the most severe power outages, and how that is changing over time.

```python
outages.groupby(['YEAR', 'POSTAL.CODE'])['SEVERE'].sum().reset_index()
```

| YEAR | POSTAL.CODE | SEVERE |
| ---: | :---------- | -----: |
| 2000 | AK          |      0 |
| 2000 | AL          |      2 |
| 2000 | AZ          |      0 |
| 2000 | CA          |      0 |
| 2000 | DE          |      0 |

<br>
<br>
Equivalently, the following map shows the states with the most number of"severe" power outages each year.

<iframe
  src="assets/choropleth.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

States with larger populations, such as California, Texas, and New York seem to dominate this space. Let's perform a transformation our data to find the number of SEVERE outages per 100,000 people.

<iframe
  src="assets/barchart_adj.html"
  width="700"
  height="400"
  frameborder="0"
></iframe>

With adjustments, large population centers like California, Texas, and New York are no longer included amongst states with the most instances of severe power outages. The data is more interpretable because now the frequency is scaled relative to population size. We see states like MI, MD, LA, and WA to continue to be amongst those with more occurrences of severe outages, while DC has made it to position 1. This might require more exploration since DC's population is very small to start, so SEVERE / population would naturally be higher.

Furthermore, states in Tornado alley have made it to leader board, so besides the outlier of DC, this scaling process has yielded interpretable results.

<span style="color: #FFCCCC;"> <b> Data Analysis Conclusion </b> </span>
To sum things up, we have analyzed potential regional risk factors in the utilities industry as well as relevant progress in the industry as a whole throughout time.
<br>
<br>

## <span style="color:#FF9999;"> Framing a Prediction Problem<span>

The prediction problem is denoted as follows.

<b><i>Predict the cause of a power outage </i></b>

This is a multiclass classification problem. We will attempt to use k-nearest neighbors algorithm and then evaluate the model's competency using F1-score. F1-score is our metric of choice when it comes to performance because the data is unbalanced (most outages are from severe weather and intentional attack), and we care about false positives since deploying the wrong resources to an outage can be substantially costly.

Knowing the cause of an outage using a predictive model helps utility companies better prepare for the corresponding damages. We will thus predict CAUSE.CATEGORY because of its real world applications (and its relevance to understanding how to build a classifier).

The feature variables used are described in the following section "Baseline Model".
These variables are static, and are therefore known at the time of the outage. We are safe to proceed.

## <span style="color:#FF9999;"> Baseline Model<span>

The feature space can be denoted as follows:

| **Variable**        | **Type**     |
| ------------------- | ------------ |
| **POSTAL.CODE**     | Nominal      |
| **CLIMATE.REGION**  | Nominal      |
| **ANOMALY.LEVEL**   | Quantitative |
| **TOTAL.CUSTOMERS** | Quantitative |
| **POPULATION**      | Quantitative |
| **POPPCT_URBAN**    | Quantitative |
| **PCT_WATER_TOT**   | Quantitative |

I used the OneHotEncoder() from scikit-learn to encode categorical variables, such as POSTAL.CODE and CLIMATE.REGION. This transformation converted these nominal variables into binary columns, allowing the model to process them effectively. I used the parameter drop='first' to avoid multicollinearity by removing the first category as a reference."

```python
   simple_preprocessing = make_column_transformer((OneHotEncoder(drop='first', handle_unknown='ignore'), ['POSTAL.CODE', 'CLIMATE.REGION']))
   simple_knn = make_pipeline(simple_preprocessing, KNeighborsClassifier())
   simple_knn.fit(X_train, y_train)
   simple_y_pred = simple_knn.predict(X_test)
   f1_score(y_test,simple_y_pred, average=None )
```

```python
    f1_simple
    array([0.        , 0.        , 0.44444444, 0.        , 0.        ,
       0.64692483, 0.18181818])
```

<iframe
  src="assets/f1_simple.html"
  width="700"
  height="400"
  frameborder="0"
></iframe>

### Performance Evaluation

This simple model fails in many ways to produce viable results and can not be consiered "good". Classes with F1-Scores of 0 indicate the model has no predictive power for those classes. There simply is not enough granularity in the feature space to distinguish outages that are not caused by severe weather. However, the model does perform moderately well in classifying severe weather events (F1-Score = 0.65). Let's see if we can improve our model using relevant transformations and hyperparameter tuning.

## <span style="color:#FF9999;"> Final Model<span>

In the final model, I kept the columns that were OneHotEncoded() and then transformed my numerical data. Because K-NN is a distance based algorithm and picks the most frequently occurring category out of the K most similar data points, having data in different scales can disrupt the similarities between observations. Thus, we need to transform using either a QuantileTransformer() or a StandardScaler(). The decision rule for these two methods is as follows:

StandardScaler(): This method is effective when the underlying features are approximately normally distributed.

QuantileTransformer(): This method is useful when the data distribution is highly skewed or not normal.

<br>
 The other critical part of improving this model was to tune relevant hyperparameters. I used GridSearchCV() to find the optimal value of k number of neighbors and the optimal distance algorithm for computating the nearest neighbors.

### Performance Improvements

```python
pipe = make_pipeline(adv_preprocessing, KNeighborsClassifier())
searcher = GridSearchCV(
        pipe,
        param_grid=hyperparams,
        scoring='f1_weighted'
    )

searcher.fit(X_train, y_train)
```

The hyperparameters that performed best were:

```python
  {'kneighborsclassifier__n_neighbors': 17,
 'kneighborsclassifier__weights': 'distance'}
```

```python
    f1_final
    array([0.08695652, 0.22222222, 0.65811966, 0.37037037, 0.4375    ,
       0.71611253, 0.3255814 ])
```

<iframe
  src="assets/f1_final.html"
  width="700"
  height="400"
  frameborder="0"
></iframe>

F1-score improved substantially, especially for the severe weather category and intentional attack.
From the confusion matrix, it appears that our model is sufficient at predicting severe weather and intentional attack. It has a tendency to mix the two up, but can somewhat decipher that these two events have some unique characteristics about them. A better model would take care of the granularity associated with the other causes, perhaps by having a more robust feature space.

<iframe
  src="assets/confusion_matrix.html"
  width="700"
  height="400"
  frameborder="0"
></iframe>

<br>
<br>
## <span style="color:#FF9999;"> Conclusion<span>
I broadly examined risk factors in the utilities industry. I then proposed a general model to classify the cause of a power outage before improving upon it sustantially through enhancing my feature space.

<!-- <iframe
  src="assets/choropleth.html"
  width="700"
  height="500"
  frameborder="0"
></iframe> -->
