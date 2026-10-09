# NYC Fare Predictor

A regression project that predicts the fare of an NYC trip for Yellow taxi, Green taxi, Uber and Lyft. The user gives a pickup zone, a dropoff zone, a date, a time and the weather, and the app shows the expected fare for all four services side by side.

Live app:  https://taxifareprediction-nyc.streamlit.app/

## Data

I did not use a ready made dataset. Everything is downloaded directly from the original public sources by the code in the notebook.

| Source | What it gives | How it is fetched |
|---|---|---|
| NYC TLC Yellow Taxi trip records | Trips with pickup and dropoff time, zones, distance and fare | Monthly parquet files from the TLC website links |
| NYC TLC Green Taxi trip records | Same layout as Yellow | Monthly parquet files from the TLC website links |
| NYC TLC High Volume For Hire records | Uber and Lyft trips | Monthly parquet files from the TLC website links |
| TLC taxi zone lookup | Zone names and boroughs | CSV from the TLC website link |
| Open-Meteo archive API | Hourly temperature, rain, snow, wind and cloud cover for NYC | API request, no key needed |

The period is all 12 months of 2024. No data files are stored in this repository because the notebook downloads them.

## How the data was prepared

The Uber and Lyft files hold around 20 million trips every month, so each month is processed on its own and the large file is read in chunks.

- Demand is counted on the full data before any sampling
- A 3 percent sample of Uber and Lyft trips is kept, then the services are balanced
- The final table has about 5.2 million trips across the four services
- The target is the base fare only, with no tips, tolls, taxes or fees, so the services are comparable

Cleaning rules

- Fare between 0 and 200 dollars
- Distance between 0 and 100 miles
- Duration between 1 and 180 minutes
- Unknown pickup and dropoff zones removed
- Rows dated outside their month removed

The 200 dollar limit was checked on the data. The January trips above it had a median distance of about 42 miles, which are out of city rides and not part of a city fare model.

## Features

| Feature | Meaning |
|---|---|
| est_distance | Median distance of the pickup and dropoff zone pair, because the app user does not enter a distance |
| avg_demand | Typical number of trips in the pickup zone for that weekday and hour, all four services together |
| avg_speed | Typical traffic speed in the pickup zone for that weekday and hour, used as a supply proxy since slow traffic means fewer free cars |
| Time features | Hour and month as sin and cos, weekend flag, rush hour flag |
| Weather | Temperature, rain, snow, wind, cloud cover, plus rain and snow flags |
| Airport flags | Pickup or dropoff at JFK, LaGuardia or Newark |
| Service | Dummy columns plus a separate distance slope for each service |

## Train and test split

The 15th to 21st of every month is held out as the test set, which is about 23 percent of the trips. Every weekday and every season appears in it. A random split would put trips from the same hour into both sets and make the score look better than it is.

Demand, speed and distance averages are built from the training days only, so nothing from the test set leaks into the model.

## Modelling

- The target is log1p of the fare because the raw fare is skewed. Skew went from 2.38 to 0.49
- Variance Inflation Factor was checked and every value was below 3, so no feature was dropped
- OLS, Ridge and Lasso were compared with the regularisation strength chosen by cross validation
- Features were standardised and kept in float64 for stable results
- With millions of rows and few features, the three models scored the same and Lasso removed no feature
- The straight line distance model made large errors on long trips, so log distance, squared distance and a log distance slope per service were added for the final Ridge model

## Results

All numbers are on the held out test weeks.

| Model | R2 | RMSE | MAE |
|---|---|---|---|
| OLS, Ridge and Lasso with straight line distance | 0.671 | 17.16 dollars | 6.79 dollars |
| Final Ridge with log and squared distance | 0.758 | 8.82 dollars | 4.93 dollars |

Average absolute error by service

| Service | Error |
|---|---|
| Yellow | 3.70 dollars |
| Green | 3.77 dollars |
| Lyft | 4.91 dollars |
| Uber | 6.59 dollars |

What stood out

- Distance is by far the strongest signal
- Weather has a very small effect on the fare in a linear model
- Taxis are easier to predict because the meter follows fixed rules
- Uber is the hardest because surge pricing is not in the features
- Trips under 20 dollars have an average error of about 2.6 dollars, while trips above 80 dollars have a much larger error

## Limitations

- Uber and Lyft are a sample and the model has no surge information
- The JFK flat fare is not captured well because the model only knows that a trip touches an airport, not the exact route
- Only 2024 is covered
- Distance in the app is an estimate from the zone pair, so trips inside one zone are rough
- Taxi fares are metered fares and Uber and Lyft fares are app prices, so they are similar but not the same thing

## The app

Built with Streamlit. The user picks pickup, dropoff, date, time, weather and temperature. The app looks up the estimated distance, demand and speed, runs the model for each service and shows the fares in a chart and a table. Green is hidden for Manhattan pickups because Green taxis cannot pick up in most of Manhattan.

Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Repository files

| File | Purpose |
|---|---|
| NYC_Fare_Prediction.ipynb | Full pipeline from download to trained model |
| app.py | Streamlit app |
| requirements.txt | Library versions |
| model.joblib | Trained Ridge model, scaler and feature list |
| pair_distance.parquet | Distance for each pickup and dropoff zone pair |
| demand_avg.parquet | Average demand lookup |
| speed_avg.parquet | Average speed lookup |
| zones.csv | Zone names and boroughs |
| weather_monthly.csv | Monthly average temperature, wind and cloud cover used by the app |

## Reproduce

1. Open NYC_Fare_Prediction.ipynb in Google Colab
2. Run all cells and allow access to Google Drive, where the intermediate files are saved
3. The first run takes about an hour, mostly downloading the Uber and Lyft files
4. If Colab disconnects, run again and finished months are skipped

## Tools

Python, pandas, NumPy, scikit-learn, statsmodels, matplotlib, Streamlit
