## Data Sources
### Raw Data
- The Flight delay data utilized was collected in 2018. 
- It consisted of 27 attributes and 7,213,446 data points.
- Please refer to the original source [here](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)

### Preprocessed Data
- The raw data file '2018.csv' was cleaned and preprocessed.
- Due to the extensive data, the data was filtered to focus on the busiest airport in the US.
- The data preprocessing and exploratory was performed using Python
- The development of classification models and their evaluation were performed using the GUI of Weka
- The preprocessed data were split into training and testing sets before being fed into Weka for computation  

The description for the preprocessed data files are outlined below:
- **`flight_data_new.csv`**: Preprocessed and cleansed data of the busiest airport in the US
- **`flight_data_new_test.arff`**: Contain 20% of the data from `flight_data_new.csv`, used for models testing, saved in .arff extension for Weka
- **`flight_data_new_test.csv`**: Contain 20% of the data from `flight_data_new.csv`, used for models testing, saved in .csv extension
- **`flight_data_new_train.arff`**: Contain 80% of the data from `flight_data_new.csv`, used for models training, saved in .arff extension for Weka
- **`flight_data_new_train.csv`**: Contain 80% of the data from `flight_data_new.csv`, used for models training, saved in .csv extension
- **`flight_data_new_train_smote.arff`**: Resampled data from `flight_data_new_train.arff` using SMOTE algorithms used for models training, saved in .arff extension for Weka
- **`flight_data_new_train_undersample.arff`**: Resampled data from `flight_data_new_train.arff` with random undersampling method used for models training, saved in .arff extension for Weka
- **`airports.csv`**: Full list of the US airport's abbreviation

