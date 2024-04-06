# Data-Camp-Supermarket-Loyalty-ML-Project
International Essentials is an international supermarket chain.

Shoppers at their supermarkets can sign up for a loyalty program that provides rewards each year to customers based on their spending. The more you spend the bigger the rewards.

The supermarket would like to be able to predict the likely amount customers in the program will spend, so they can estimate the cost of the rewards.

This will help them to predict the likely profit at the end of the year.

# Task 1
Before you fit any models, you will need to make sure the data is clean.
The table below shows what the data should look like.
Create a cleaned version of the dataframe.

1- You should start with the data in the file "loyalty.csv".

2- Your output should be a dataframe named clean_data.




# Task 2
The team at International Essentials have told you that they have always believed that the number of years in the loyalty scheme is the biggest driver of spend.

Producing a table showing the difference in the average spend by number of years in the loyalty programme along with the variance to investigate this question for the team.

You should start with the data in the file 'loyalty.csv'.

1- Your output should be a data frame named spend_by_years.

2- It should include the three columns loyalty_years, avg_spend, var_spend.

3- Your answers should be rounded to 2 decimal places.


# Task 3
Fit a baseline model to predict the spend over the year for each customer.

1- Fit your model using the data contained in “train.csv”


2- Use “test.csv” to predict new values based on your model. You must return a dataframe named base_result, that includes customer_id and spend. The spend column must be your predicted values.


# Task 4
Fit a comparison model to predict the sale price of a house.

1- Fit your model using the data contained in “train.csv”


2- Use “test.csv” to predict new values based on your model. You must return a dataframe named compare_result, that includes customer_id and spend. The spend column must be your predicted values.
