# ec2-spot-price-forecaster

## Requirements
* Python 3.6 or above - it doesn't matter conda env or virtenv
* All the packages described in requirements file
* Decompress dataset file

```sh
# inside of dataset folder
unzip eu-west-1.csv.zip
```

## How does it run?
Just run the arima or lstm script with no arguments

```sh
# inside of src folder
python arima_prediction.py
python lstm_prediction.py
```

## Some usefull informaton
* Predicting Amazon Spot Prices with LSTM Networks (from papers folder)
* https://www.alkaline-ml.com/pmdarima/auto_examples/example_simple_fit.html
