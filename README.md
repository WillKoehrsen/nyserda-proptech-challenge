# nyserda-proptech-challenge

Work for the [NYSERDA Proptech Data Challenge](https://www.proptechchallenge.com/nyserda-tenant-energy-data)

## Problem Statements

1. What is your forecasted consumption across all 18 tenant usage meters for the 24 hours of 8/31/20 in 15 minute intervals (1728 predictions)?


2. How correlated are building-wide occupancy and tenant consumption?


3. What is the mean absolute error for your model?


4. What feature(s)/predictor(s) were most important in determining energy efficiency?


5. What is the most energy-efficient occupancy level as a percentage of max occupancy provided (i.e., occupancy on 2/10/20)?


6. What else, if anything, can be concluded from your model?

Energy efficiency is highest with the greatest reduction in occupancy. Even a few tenants in the building drastically increases the consumption indicating the need for coordination between building engineers and tenants. If building engineers know the floors that are occupied, they may be able to increase efficiency by selecting shutting off equipment.

7. What other information, if any, would you need to better your model?

Occupancy corresponding to each floor would be extremely useful because this data could be directly linked to the meter for the floor/area. Another helpful piece of information would be the lease obligations, that is, the time at which the building must be within a comfortable temperature range. This would help identify when the building should be starting up, and could be combined with internal floor temperatures to determine if the floor is conditioned too early. Were the floor reaching temperature too early, this would be an opportunity to save energy by starting the building's equipment later.

Any internal time-series data from the Building Management System (BMS) could potentially improve this model, or allow us to construct other models for accurately forecasting consumption and to identify efficient operation.