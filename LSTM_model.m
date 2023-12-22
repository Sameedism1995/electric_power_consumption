% Loading the datasets
clearvars;close all;clc
powerUsageData = readtable('power_usage_2016_to_2020.csv');
weatherData = readtable('weather_2016_2020_daily.csv');

% Converting dates to datetime format
powerUsageData.StartDate = datetime(powerUsageData.StartDate, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
weatherData.Date = datetime(weatherData.Date, 'InputFormat', 'yyyy-MM-dd');

% Indeces to exclude
excludeYears = [2016,2020];
powerExcludeInd = ismember(year(powerUsageData.StartDate),excludeYears);
powerUsageData_orig = powerUsageData;
powerUsageData = powerUsageData(~powerExcludeInd,:);

weatherExcludeInd = ismember(year(weatherData.Date),excludeYears);
weatherData_orig = weatherData;
weatherData = weatherData(~weatherExcludeInd,:);

% Aggregating power usage data to daily using manual computation
powerUsageData.StartDate = dateshift(powerUsageData.StartDate, 'start', 'day');
uniqueDates = unique(powerUsageData.StartDate);
totalDailyPowerUsage = zeros(size(uniqueDates));
for i = 1:length(uniqueDates)
    totalDailyPowerUsage(i) = sum(powerUsageData{powerUsageData.StartDate == uniqueDates(i), 2});
end
dailyPowerUsage = table(uniqueDates, totalDailyPowerUsage, 'VariableNames', {'StartDate', 'TotalDailyPowerUsage'});

% Now merging the datasets
mergedData = outerjoin(dailyPowerUsage, weatherData, 'LeftKeys', 'StartDate', 'RightKeys', 'Date', ...
                       'MergeKeys', true);

% Now handling missing data - removing rows with NaN values
mergedData = rmmissing(mergedData);

% Feature selection
features = mergedData(:, {'TotalDailyPowerUsage', 'Temp_avg', 'Press_avg','Dew_avg'});

% Normalizing the features
mu = mean(features{:,:});
sig = std(features{:,:});
features{:,:} = (features{:,:} - mu) ./ sig;

% Shifting the target variable to create the 'previous day power usage' feature
features.previousDayPowerUsage = [nan; features{1:end-1, 1}];

% Removing the first row with NaN
features(1, :) = [];

% Preparing data for LSTM
X = features{:, 2:end};
y = features{:, 1};

% Reshaping data for LSTM
numObservations = size(X, 1);
numFeatures = size(X, 2);
numTimeSteps = 1;
X = reshape(X', [numFeatures , numObservations]);
y = reshape(y, [1, numObservations]);
% Spliting data into training and test sets
numTimeStepsTrain = floor(0.8 * numObservations);
XTrain = X(:, 1:numTimeStepsTrain);
yTrain = y(:, 1:numTimeStepsTrain);
XTest = X(:, numTimeStepsTrain+1:end);
yTest = y(:, numTimeStepsTrain+1:end);
dateTrain = mergedData.StartDate_Date(1:numTimeStepsTrain);
dateTest = mergedData.StartDate_Date(numTimeStepsTrain+2:end);
[dateTest,idx] = sort([dateTest],'ascend');

% Defining LSTM network architecture
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50, 'OutputMode', 'sequence')
    dropoutLayer(0.1)
    fullyConnectedLayer(1)
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.05, ...
    'MiniBatchSize',128/2, ...
    'SequenceLength',"shortest", ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Training the network
net = trainNetwork(XTrain, yTrain, layers, options);

% Predicting and evaluate the model
YPred = predict(net, XTest);

% Calculating RMSE
rmse = sqrt(mean((YPred - yTest).^2))

% Plots of the prediction:
figure; 
nexttile;
plot(dateTest,yTest);
hold on 
plot(dateTest,YPred);
title("Test partition one-ahead predictions");
legend('True values','Estimations')