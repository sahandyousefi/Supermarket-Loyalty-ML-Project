# Data-Camp-Supermarket-Loyalty-ML-Project
International Essentials is an international supermarket chain.

Shoppers at their supermarkets can sign up for a loyalty program that provides rewards each year to customers based on their spending. The more you spend the bigger the rewards.

The supermarket would like to be able to predict the likely amount customers in the program will spend, so they can estimate the cost of the rewards.

This will help them to predict the likely profit at the end of the year.

Data
The dataset contains records of customers for their last full year of the loyalty program.

Column Name	Criteria
customer_id	Unique identifier for the customer.
Missing values are not possible due to the database structure.
spend	Continuous.
The total spend of the customer in their last full year. This can be any positive value to two decimal places.
Missing values should be replaced with 0.
first_month	Continuous.
The amount spent by the customer in their first month of the year. This can be any positive value, rounded to two decimal places.
Missing values should be replaced with 0.
items_in_first_month	Discrete.
The number of items purchased in the first month. Any integer value greater than or equal to zero.
Missing values should be replaced by 0.
region	Nominal.
The geographic region that the customer is based in. One of four values Americas, Asia/Pacific, Europe, Middle East/Africa.
Missing values should be replaced with "Unknown".
loyalty_years	Oridinal.
The number of years the customer has been a part of the loyalty program. One of five ordered categories, '0-1', '1-3', '3-5', '5-10', '10+'.
Missing values should be replaced with '0-1'.
joining_month	Nominal.
The month the customer joined the loyalty program. One of 12 values "Jan", "Feb", "Mar", "Apr", etc.
Missing values should be replaced with "Unknown".
promotion	Nominal.
Did the customer join the loyalty program as part of a promotion? Either 'Yes' or 'No'.
Missing values should be replaced with 'No'.
