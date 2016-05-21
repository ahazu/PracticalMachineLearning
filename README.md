# Prediction Assignment - Practical Machine Learning
Per Rynning  
19 mai 2016  

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> 

The required R packages to reproduce this analysis are:

```r
library(knitr)
library(caret)
library(doParallel)
registerDoParallel()
```

### Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

Unless the data is available in the current working directory, the files will be automaticallt downloaded.


```r
## Downloads training data if not available in the current folder
if(!file.exists("pml-training.csv")){
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                  destfile = "pml-training.csv")
    
}

## Downloads testing data if not available in the current folder
if(!file.exists("pml-testing.csv")){
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                  destfile = "pml-testing.csv")
    
}
```

The data was downloaded 19.05.2016 - 09:50. The data may change in the future.

We'll read the training and testing datasets. We will however refer to to test data as validation data from this point on, as we will pick out some additional testing data from the training data.


```r
training <- read.csv(file = "pml-training.csv", header = TRUE)

validation <- read.csv(file = "pml-testing.csv", header = TRUE)
```

##Exploratory Data Analysis


```r
str(training)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : Factor w/ 397 levels "","-0.016850",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_belt     : Factor w/ 317 levels "","-0.021887",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt      : Factor w/ 395 levels "","-0.003095",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_belt.1    : Factor w/ 338 levels "","-0.005928",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_belt       : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_belt            : Factor w/ 68 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : Factor w/ 4 levels "","#DIV/0!","0.00",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ var_total_accel_belt    : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_belt          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_belt            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y            : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z            : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x            : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
##  $ accel_belt_y            : int  4 4 5 3 2 4 3 4 2 4 ...
##  $ accel_belt_z            : int  22 22 23 21 24 21 21 21 24 22 ...
##  $ magnet_belt_x           : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y           : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z           : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm                : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm               : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm                 : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm         : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ var_accel_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ var_yaw_arm             : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_y             : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
##  $ gyros_arm_z             : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_x             : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
##  $ accel_arm_y             : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z             : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x            : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_y            : int  337 337 344 344 337 342 336 338 341 334 ...
##  $ magnet_arm_z            : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ kurtosis_roll_arm       : Factor w/ 330 levels "","-0.02438",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_arm      : Factor w/ 328 levels "","-0.00484",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_arm        : Factor w/ 395 levels "","-0.01548",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_arm       : Factor w/ 331 levels "","-0.00051",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_arm      : Factor w/ 328 levels "","-0.00184",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_arm        : Factor w/ 395 levels "","-0.00311",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_roll_arm            : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_arm           : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_arm             : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell          : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell            : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ kurtosis_roll_dumbbell  : Factor w/ 398 levels "","-0.0035","-0.0073",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_picth_dumbbell : Factor w/ 401 levels "","-0.0163","-0.0233",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ kurtosis_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_roll_dumbbell  : Factor w/ 401 levels "","-0.0082","-0.0096",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_pitch_dumbbell : Factor w/ 402 levels "","-0.0053","-0.0084",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ skewness_yaw_dumbbell   : Factor w/ 2 levels "","#DIV/0!": 1 1 1 1 1 1 1 1 1 1 ...
##  $ max_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ min_roll_dumbbell       : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : num  NA NA NA NA NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : Factor w/ 73 levels "","-0.1","-0.2",..: 1 1 1 1 1 1 1 1 1 1 ...
##  $ amplitude_roll_dumbbell : num  NA NA NA NA NA NA NA NA NA NA ...
##   [list output truncated]
```

```r
str(validation)
```

```
## 'data.frame':	20 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 6 5 5 1 4 5 5 5 2 3 ...
##  $ raw_timestamp_part_1    : int  1323095002 1322673067 1322673075 1322832789 1322489635 1322673149 1322673128 1322673076 1323084240 1322837822 ...
##  $ raw_timestamp_part_2    : int  868349 778725 342967 560311 814776 510661 766645 54671 916313 384285 ...
##  $ cvtd_timestamp          : Factor w/ 11 levels "02/12/2011 13:33",..: 5 10 10 1 6 11 11 10 3 2 ...
##  $ new_window              : Factor w/ 1 level "no": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  74 431 439 194 235 504 485 440 323 664 ...
##  $ roll_belt               : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt              : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt                : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
##  $ total_accel_belt        : int  20 4 5 17 3 4 4 4 4 18 ...
##  $ kurtosis_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_belt     : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt      : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_belt.1    : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_belt       : logi  NA NA NA NA NA NA ...
##  $ max_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ max_picth_belt          : logi  NA NA NA NA NA NA ...
##  $ max_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ min_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ min_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ min_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_belt     : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_belt    : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_belt      : logi  NA NA NA NA NA NA ...
##  $ var_total_accel_belt    : logi  NA NA NA NA NA NA ...
##  $ avg_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_belt        : logi  NA NA NA NA NA NA ...
##  $ var_roll_belt           : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_belt       : logi  NA NA NA NA NA NA ...
##  $ var_pitch_belt          : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_belt         : logi  NA NA NA NA NA NA ...
##  $ var_yaw_belt            : logi  NA NA NA NA NA NA ...
##  $ gyros_belt_x            : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
##  $ gyros_belt_y            : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
##  $ gyros_belt_z            : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
##  $ accel_belt_x            : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
##  $ accel_belt_y            : int  69 11 -1 45 4 -16 2 -2 1 63 ...
##  $ accel_belt_z            : int  -179 39 49 -156 27 38 35 42 32 -158 ...
##  $ magnet_belt_x           : int  -13 43 29 169 33 31 50 39 -6 10 ...
##  $ magnet_belt_y           : int  581 636 631 608 566 638 622 635 600 601 ...
##  $ magnet_belt_z           : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
##  $ roll_arm                : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
##  $ pitch_arm               : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
##  $ yaw_arm                 : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
##  $ total_accel_arm         : int  10 38 44 25 29 14 15 22 34 32 ...
##  $ var_accel_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ stddev_roll_arm         : logi  NA NA NA NA NA NA ...
##  $ var_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ avg_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ stddev_pitch_arm        : logi  NA NA NA NA NA NA ...
##  $ var_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ avg_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ stddev_yaw_arm          : logi  NA NA NA NA NA NA ...
##  $ var_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ gyros_arm_x             : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
##  $ gyros_arm_y             : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
##  $ gyros_arm_z             : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
##  $ accel_arm_x             : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
##  $ accel_arm_y             : int  38 215 245 -57 200 130 79 175 111 -42 ...
##  $ accel_arm_z             : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
##  $ magnet_arm_x            : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
##  $ magnet_arm_y            : int  385 447 474 257 275 176 15 215 335 294 ...
##  $ magnet_arm_z            : int  481 434 413 633 617 516 217 385 520 493 ...
##  $ kurtosis_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_arm      : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_arm       : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_arm      : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_arm        : logi  NA NA NA NA NA NA ...
##  $ max_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ max_picth_arm           : logi  NA NA NA NA NA NA ...
##  $ max_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ min_roll_arm            : logi  NA NA NA NA NA NA ...
##  $ min_pitch_arm           : logi  NA NA NA NA NA NA ...
##  $ min_yaw_arm             : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_arm      : logi  NA NA NA NA NA NA ...
##  $ amplitude_pitch_arm     : logi  NA NA NA NA NA NA ...
##  $ amplitude_yaw_arm       : logi  NA NA NA NA NA NA ...
##  $ roll_dumbbell           : num  -17.7 54.5 57.1 43.1 -101.4 ...
##  $ pitch_dumbbell          : num  25 -53.7 -51.4 -30 -53.4 ...
##  $ yaw_dumbbell            : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
##  $ kurtosis_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ kurtosis_picth_dumbbell : logi  NA NA NA NA NA NA ...
##  $ kurtosis_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ skewness_roll_dumbbell  : logi  NA NA NA NA NA NA ...
##  $ skewness_pitch_dumbbell : logi  NA NA NA NA NA NA ...
##  $ skewness_yaw_dumbbell   : logi  NA NA NA NA NA NA ...
##  $ max_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ max_picth_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ max_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ min_roll_dumbbell       : logi  NA NA NA NA NA NA ...
##  $ min_pitch_dumbbell      : logi  NA NA NA NA NA NA ...
##  $ min_yaw_dumbbell        : logi  NA NA NA NA NA NA ...
##  $ amplitude_roll_dumbbell : logi  NA NA NA NA NA NA ...
##   [list output truncated]
```

Let's have a look at the training data and see if there are any Zero, or Near-zero covariates:


```r
nzv <- nearZeroVar(training, 
                   saveMetrics=TRUE)
nzv
```

```
##                            freqRatio percentUnique zeroVar   nzv
## X                           1.000000  100.00000000   FALSE FALSE
## user_name                   1.100679    0.03057792   FALSE FALSE
## raw_timestamp_part_1        1.000000    4.26562022   FALSE FALSE
## raw_timestamp_part_2        1.000000   85.53154622   FALSE FALSE
## cvtd_timestamp              1.000668    0.10192641   FALSE FALSE
## new_window                 47.330049    0.01019264   FALSE  TRUE
## num_window                  1.000000    4.37264295   FALSE FALSE
## roll_belt                   1.101904    6.77810621   FALSE FALSE
## pitch_belt                  1.036082    9.37722964   FALSE FALSE
## yaw_belt                    1.058480    9.97349913   FALSE FALSE
## total_accel_belt            1.063160    0.14779329   FALSE FALSE
## kurtosis_roll_belt       1921.600000    2.02323922   FALSE  TRUE
## kurtosis_picth_belt       600.500000    1.61553358   FALSE  TRUE
## kurtosis_yaw_belt          47.330049    0.01019264   FALSE  TRUE
## skewness_roll_belt       2135.111111    2.01304658   FALSE  TRUE
## skewness_roll_belt.1      600.500000    1.72255631   FALSE  TRUE
## skewness_yaw_belt          47.330049    0.01019264   FALSE  TRUE
## max_roll_belt               1.000000    0.99378249   FALSE FALSE
## max_picth_belt              1.538462    0.11211905   FALSE FALSE
## max_yaw_belt              640.533333    0.34654979   FALSE  TRUE
## min_roll_belt               1.000000    0.93772296   FALSE FALSE
## min_pitch_belt              2.192308    0.08154113   FALSE FALSE
## min_yaw_belt              640.533333    0.34654979   FALSE  TRUE
## amplitude_roll_belt         1.290323    0.75425543   FALSE FALSE
## amplitude_pitch_belt        3.042254    0.06625217   FALSE FALSE
## amplitude_yaw_belt         50.041667    0.02038528   FALSE  TRUE
## var_total_accel_belt        1.426829    0.33126083   FALSE FALSE
## avg_roll_belt               1.066667    0.97339721   FALSE FALSE
## stddev_roll_belt            1.039216    0.35164611   FALSE FALSE
## var_roll_belt               1.615385    0.48924676   FALSE FALSE
## avg_pitch_belt              1.375000    1.09061258   FALSE FALSE
## stddev_pitch_belt           1.161290    0.21914178   FALSE FALSE
## var_pitch_belt              1.307692    0.32106819   FALSE FALSE
## avg_yaw_belt                1.200000    1.22311691   FALSE FALSE
## stddev_yaw_belt             1.693878    0.29558659   FALSE FALSE
## var_yaw_belt                1.500000    0.73896647   FALSE FALSE
## gyros_belt_x                1.058651    0.71348486   FALSE FALSE
## gyros_belt_y                1.144000    0.35164611   FALSE FALSE
## gyros_belt_z                1.066214    0.86127816   FALSE FALSE
## accel_belt_x                1.055412    0.83579655   FALSE FALSE
## accel_belt_y                1.113725    0.72877383   FALSE FALSE
## accel_belt_z                1.078767    1.52379982   FALSE FALSE
## magnet_belt_x               1.090141    1.66649679   FALSE FALSE
## magnet_belt_y               1.099688    1.51870350   FALSE FALSE
## magnet_belt_z               1.006369    2.32901845   FALSE FALSE
## roll_arm                   52.338462   13.52563449   FALSE FALSE
## pitch_arm                  87.256410   15.73234125   FALSE FALSE
## yaw_arm                    33.029126   14.65701763   FALSE FALSE
## total_accel_arm             1.024526    0.33635715   FALSE FALSE
## var_accel_arm               5.500000    2.01304658   FALSE FALSE
## avg_roll_arm               77.000000    1.68178575   FALSE  TRUE
## stddev_roll_arm            77.000000    1.68178575   FALSE  TRUE
## var_roll_arm               77.000000    1.68178575   FALSE  TRUE
## avg_pitch_arm              77.000000    1.68178575   FALSE  TRUE
## stddev_pitch_arm           77.000000    1.68178575   FALSE  TRUE
## var_pitch_arm              77.000000    1.68178575   FALSE  TRUE
## avg_yaw_arm                77.000000    1.68178575   FALSE  TRUE
## stddev_yaw_arm             80.000000    1.66649679   FALSE  TRUE
## var_yaw_arm                80.000000    1.66649679   FALSE  TRUE
## gyros_arm_x                 1.015504    3.27693405   FALSE FALSE
## gyros_arm_y                 1.454369    1.91621649   FALSE FALSE
## gyros_arm_z                 1.110687    1.26388747   FALSE FALSE
## accel_arm_x                 1.017341    3.95984099   FALSE FALSE
## accel_arm_y                 1.140187    2.73672409   FALSE FALSE
## accel_arm_z                 1.128000    4.03628580   FALSE FALSE
## magnet_arm_x                1.000000    6.82397309   FALSE FALSE
## magnet_arm_y                1.056818    4.44399144   FALSE FALSE
## magnet_arm_z                1.036364    6.44684538   FALSE FALSE
## kurtosis_roll_arm         246.358974    1.68178575   FALSE  TRUE
## kurtosis_picth_arm        240.200000    1.67159311   FALSE  TRUE
## kurtosis_yaw_arm         1746.909091    2.01304658   FALSE  TRUE
## skewness_roll_arm         249.558442    1.68688207   FALSE  TRUE
## skewness_pitch_arm        240.200000    1.67159311   FALSE  TRUE
## skewness_yaw_arm         1746.909091    2.01304658   FALSE  TRUE
## max_roll_arm               25.666667    1.47793293   FALSE  TRUE
## max_picth_arm              12.833333    1.34033228   FALSE FALSE
## max_yaw_arm                 1.227273    0.25991234   FALSE FALSE
## min_roll_arm               19.250000    1.41677709   FALSE  TRUE
## min_pitch_arm              19.250000    1.47793293   FALSE  TRUE
## min_yaw_arm                 1.000000    0.19366018   FALSE FALSE
## amplitude_roll_arm         25.666667    1.55947406   FALSE  TRUE
## amplitude_pitch_arm        20.000000    1.49831821   FALSE  TRUE
## amplitude_yaw_arm           1.037037    0.25991234   FALSE FALSE
## roll_dumbbell               1.022388   84.20650290   FALSE FALSE
## pitch_dumbbell              2.277372   81.74498012   FALSE FALSE
## yaw_dumbbell                1.132231   83.48282540   FALSE FALSE
## kurtosis_roll_dumbbell   3843.200000    2.02833554   FALSE  TRUE
## kurtosis_picth_dumbbell  9608.000000    2.04362450   FALSE  TRUE
## kurtosis_yaw_dumbbell      47.330049    0.01019264   FALSE  TRUE
## skewness_roll_dumbbell   4804.000000    2.04362450   FALSE  TRUE
## skewness_pitch_dumbbell  9608.000000    2.04872082   FALSE  TRUE
## skewness_yaw_dumbbell      47.330049    0.01019264   FALSE  TRUE
## max_roll_dumbbell           1.000000    1.72255631   FALSE FALSE
## max_picth_dumbbell          1.333333    1.72765263   FALSE FALSE
## max_yaw_dumbbell          960.800000    0.37203139   FALSE  TRUE
## min_roll_dumbbell           1.000000    1.69197839   FALSE FALSE
## min_pitch_dumbbell          1.666667    1.81429008   FALSE FALSE
## min_yaw_dumbbell          960.800000    0.37203139   FALSE  TRUE
## amplitude_roll_dumbbell     8.000000    1.97227602   FALSE FALSE
## amplitude_pitch_dumbbell    8.000000    1.95189073   FALSE FALSE
## amplitude_yaw_dumbbell     47.920200    0.01528896   FALSE  TRUE
## total_accel_dumbbell        1.072634    0.21914178   FALSE FALSE
## var_accel_dumbbell          6.000000    1.95698706   FALSE FALSE
## avg_roll_dumbbell           1.000000    2.02323922   FALSE FALSE
## stddev_roll_dumbbell       16.000000    1.99266130   FALSE FALSE
## var_roll_dumbbell          16.000000    1.99266130   FALSE FALSE
## avg_pitch_dumbbell          1.000000    2.02323922   FALSE FALSE
## stddev_pitch_dumbbell      16.000000    1.99266130   FALSE FALSE
## var_pitch_dumbbell         16.000000    1.99266130   FALSE FALSE
## avg_yaw_dumbbell            1.000000    2.02323922   FALSE FALSE
## stddev_yaw_dumbbell        16.000000    1.99266130   FALSE FALSE
## var_yaw_dumbbell           16.000000    1.99266130   FALSE FALSE
## gyros_dumbbell_x            1.003268    1.22821323   FALSE FALSE
## gyros_dumbbell_y            1.264957    1.41677709   FALSE FALSE
## gyros_dumbbell_z            1.060100    1.04984201   FALSE FALSE
## accel_dumbbell_x            1.018018    2.16593619   FALSE FALSE
## accel_dumbbell_y            1.053061    2.37488533   FALSE FALSE
## accel_dumbbell_z            1.133333    2.08949139   FALSE FALSE
## magnet_dumbbell_x           1.098266    5.74864948   FALSE FALSE
## magnet_dumbbell_y           1.197740    4.30129447   FALSE FALSE
## magnet_dumbbell_z           1.020833    3.44511263   FALSE FALSE
## roll_forearm               11.589286   11.08959331   FALSE FALSE
## pitch_forearm              65.983051   14.85577413   FALSE FALSE
## yaw_forearm                15.322835   10.14677403   FALSE FALSE
## kurtosis_roll_forearm     228.761905    1.64101519   FALSE  TRUE
## kurtosis_picth_forearm    226.070588    1.64611151   FALSE  TRUE
## kurtosis_yaw_forearm       47.330049    0.01019264   FALSE  TRUE
## skewness_roll_forearm     231.518072    1.64611151   FALSE  TRUE
## skewness_pitch_forearm    226.070588    1.62572623   FALSE  TRUE
## skewness_yaw_forearm       47.330049    0.01019264   FALSE  TRUE
## max_roll_forearm           27.666667    1.38110284   FALSE  TRUE
## max_picth_forearm           2.964286    0.78992967   FALSE FALSE
## max_yaw_forearm           228.761905    0.22933442   FALSE  TRUE
## min_roll_forearm           27.666667    1.37091020   FALSE  TRUE
## min_pitch_forearm           2.862069    0.87147080   FALSE FALSE
## min_yaw_forearm           228.761905    0.22933442   FALSE  TRUE
## amplitude_roll_forearm     20.750000    1.49322189   FALSE  TRUE
## amplitude_pitch_forearm     3.269231    0.93262664   FALSE FALSE
## amplitude_yaw_forearm      59.677019    0.01528896   FALSE  TRUE
## total_accel_forearm         1.128928    0.35674243   FALSE FALSE
## var_accel_forearm           3.500000    2.03343186   FALSE FALSE
## avg_roll_forearm           27.666667    1.64101519   FALSE  TRUE
## stddev_roll_forearm        87.000000    1.63082255   FALSE  TRUE
## var_roll_forearm           87.000000    1.63082255   FALSE  TRUE
## avg_pitch_forearm          83.000000    1.65120783   FALSE  TRUE
## stddev_pitch_forearm       41.500000    1.64611151   FALSE  TRUE
## var_pitch_forearm          83.000000    1.65120783   FALSE  TRUE
## avg_yaw_forearm            83.000000    1.65120783   FALSE  TRUE
## stddev_yaw_forearm         85.000000    1.64101519   FALSE  TRUE
## var_yaw_forearm            85.000000    1.64101519   FALSE  TRUE
## gyros_forearm_x             1.059273    1.51870350   FALSE FALSE
## gyros_forearm_y             1.036554    3.77637346   FALSE FALSE
## gyros_forearm_z             1.122917    1.56457038   FALSE FALSE
## accel_forearm_x             1.126437    4.04647844   FALSE FALSE
## accel_forearm_y             1.059406    5.11160942   FALSE FALSE
## accel_forearm_z             1.006250    2.95586586   FALSE FALSE
## magnet_forearm_x            1.012346    7.76679238   FALSE FALSE
## magnet_forearm_y            1.246914    9.54031189   FALSE FALSE
## magnet_forearm_z            1.000000    8.57710733   FALSE FALSE
## classe                      1.469581    0.02548160   FALSE FALSE
```

##Data Processing

###Near Zero Values
From the nzv column above we can see that there are several variables that we want to throw out, as these are not useful when we want to construct our prediction model.

We'll also remove the same variables from the test data.


```r
training.noNZV <- training[,!nzv$nzv]

validation.noNZV <- validation[,!nzv$nzv]
```

###Unnecessary columns
From looking at the 6 first column names of the training set, we can see that there is data that we'd like to exclude. 


```r
names(training.noNZV[,1:6])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"
```

```r
names(validation.noNZV[,1:6])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"
```
This data is probably important for analysing other things, but may adversely affect our prediction model. Away they go!


```r
training.noNZV <- training.noNZV[,-c(1:6)]
validation.noNZV <- validation.noNZV[,-c(1:6)]
```

###High NA columns
From briefly looking at the data earlier, we could see that there may be a lot of NAs in the data. We will exclude any column that has >70% NAs from both the training and validation sets.


```r
# Finds columns where NAs account for more than 70%
manyNAs <- colMeans(is.na(training.noNZV))>.70

training.noNZV.NAred <- training.noNZV[,!manyNAs]
validation.noNZV.NAred <- validation.noNZV[,!manyNAs]
```

###Splitting the data into training- and testing datasets
As mentioned earlier, we will separate out a portion of the training data as a test-set. This is because we want to be able to measure how accurate our model is. This will be hard to do with a test dataset with only 20 entries. We will use 70% of the data for training, and the remaining 30% for testing.


```r
inTrain <- createDataPartition(y=training.noNZV.NAred$classe, 
                               p=0.7, 
                               list=FALSE)

# We'll call the datasets .ready, as they are ready to be used in the training process
training.ready <- training.noNZV.NAred[inTrain,]
testing.ready <- training.noNZV.NAred[-inTrain,]
```

#Prediction models

##Prediction with trees
The first model we'll create is a general tree. 


```r
rpart.model <- train(classe ~., 
                     data=training.ready, 
                     method="rpart")
```

```
## Loading required package: rpart
```

Let's see how accurate this model is by testing it versus the testing dataset we separated out:


```r
confusionMatrix(testing.ready$classe, 
                predict(rpart.model,
                        testing.ready))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1495   24  146    0    9
##          B  494  365  280    0    0
##          C  472   37  517    0    0
##          D  435  160  369    0    0
##          E  140  146  296    0  500
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4889         
##                  95% CI : (0.476, 0.5017)
##     No Information Rate : 0.5159         
##     P-Value [Acc > NIR] : 1              
##                                          
##                   Kappa : 0.3324         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4924  0.49863  0.32152       NA  0.98232
## Specificity            0.9372  0.84980  0.88099   0.8362  0.89174
## Pos Pred Value         0.8931  0.32046  0.50390       NA  0.46211
## Neg Pred Value         0.6341  0.92267  0.77547       NA  0.99813
## Prevalence             0.5159  0.12438  0.27324   0.0000  0.08649
## Detection Rate         0.2540  0.06202  0.08785   0.0000  0.08496
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7148  0.67422  0.60125       NA  0.93703
```

49.2% accuracy... Worse than a coin toss. We'll have to do better than that!

##Random forest
The second model we'll create is a random forest model.


```r
rf.model <- train(classe ~., 
                  data=training.ready, 
                  method="rf")
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

Let's have a look at how accurate this model is:


```r
confusionMatrix(testing.ready$classe, 
                predict(rf.model,
                        testing.ready))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    0    0    1
##          B    5 1133    1    0    0
##          C    0    6 1020    0    0
##          D    0    0    7  957    0
##          E    0    0    0    6 1076
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9954         
##                  95% CI : (0.9933, 0.997)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9942         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9939   0.9922   0.9938   0.9991
## Specificity            0.9995   0.9987   0.9988   0.9986   0.9988
## Pos Pred Value         0.9988   0.9947   0.9942   0.9927   0.9945
## Neg Pred Value         0.9988   0.9985   0.9984   0.9988   0.9998
## Prevalence             0.2850   0.1937   0.1747   0.1636   0.1830
## Detection Rate         0.2841   0.1925   0.1733   0.1626   0.1828
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9983   0.9963   0.9955   0.9962   0.9989
```

99.2% accuracy seems acceptable. Let's use this model to find the actions from the validation dataset:


```r
predict(rf.model, 
        validation.noNZV.NAred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
