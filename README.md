# Flight Delays Prediction Using Machine Learning Approach

## Introduction

## Problem Statement
The ability to predict a delay in flight can be helpful for all parties, including airlines and passengers. This study explores the method of predicting flight delay by classifying a specific flight as either delay or no delay. From the initial review, the flight delay dataset is skewed. It is expected since most airlines usually have more non-delayed flights than delayed ones. Hence, this study compares different methods to deal with an imbalanced dataset by training a flight delay prediction model.

## Objectives
The objectives of this study are:
1.	To identify the attributes that affect flight delay. 
2.	To develop machine learning models that classify flight outcomes (either delayed or not delayed) with selected features.
3.	To evaluate the performance of different machine learning models.

## Motivation
Flight delays are significant concerns in aviation industries, leading to revenue loss, fuel loss, and customer dissatisfaction. It creates fear among passengers taking a connecting flight, whereby the delay from the first flight could potentially cause them to miss the subsequent flight. Therefore, this scenario is a factor of motivation for this study. With a reliable method to predict flight delays, the event mentioned in the previous context could either be prevented or better managed.

## Data Preprocessing
To facilitate the modeling process, the only flight data that was considered and included was the data from the busiest airports since they contained the most significant number of schedules for arrival flights in the U.S. Data cleansing was performed on the name of flight carrier, origin airport and destination airport as the abbreviation of IATA code was used. Attributes with more than 50% of missing values that did not provide helpful information to this analysis were dropped—unrelated attributes such as attributes that recorded the outcome of canceled flights and diverted flights were also removed. Since our main objective was to predict flight delay, attributes relating to canceled flights were eliminated. Instances with missing values were removed as the number of missing values was less than 1%, which was relatively small. 

For classification purposes, a binary attribute, namely "flight delay," was added to the record status of the flight. The duration between the flights taking off and the wheels off the ground, as well as flight on land and wheels on land, were derived as this provided information about the actual duration of these activities. Information about a month, day, and day of the week was transformed from the actual flight date. Before modeling, all categorical attributes such as destination airports, day of the week, flight carrier, and flight delay factors were converted to numerical variables via one hot encoding method. One dummy variable would be created for every object in the categorical variable. If the category is presented, the value would be denoted as one. Otherwise, the value would be denoted as zero.


