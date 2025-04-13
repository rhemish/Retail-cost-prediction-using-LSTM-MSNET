# LSTM-MSNET
Implementation of LSTM-MSNet: Leveraging Forecasts on Sets of Related Time Series with Multiple Seasonal Patterns by kasungayan

# About the Dataset:
For training use Quarterly-train.csv : 
- https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset?select=Quarterly-train.csv

For testing use Quarterly-test.csv :
- https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset?select=Quarterly-test.csv

This implementation utilized a quarterly time series dataset, differing from the original paper’s hourly data. Due to computational constraints, a subset of 1,000 rows was used for training and testing. This subset has been uploaded to the repository for reproducibility.

## Overview﻿
 Monitoring and evaluating our clients' business data in real time to identify any events that could affect their revenue is one of our biggest difficulties. 
 Autonomous forecasting is a subset of this challenge which predicts demand and business outcomes to optimize client operations for all potential future scenarios. 
 The article that aims to address the Autonomous Forecasting Problem Statement is highlighted in this Problem Statement:
 LSTM-MSNet Overview
﻿
In order to anticipate time series having different seasonal patterns, the author of this research suggests a unified prediction framework based on decomposition called Long Short-Term Memory Multi-Seasonal Net (LSTM-MSNet). 
 The study addresses Time Series Forecasting and suggests a technique that draws influence from the fields of Deep Learning and Statistics.   The paper's author tests their approach on three public time series datasets and shows encouraging results on each.

## How to Run
1.	Open the Jupyter Notebook file in the repository.
2.	Ensure the dataset files (Quarterly-train.csv and Quarterly-test.csv) are placed in the appropriate directory as mentioned in the code.
3.	Run the notebook or script cells sequentially:
a.	Preprocessing steps
b.	Model training and validation
c.	Testing and results visualization
 ﻿
## Breaking Down The Paper
https://storage.googleapis.com/slite-api-files-production/files/0667aeba-8911-41ae-99d3-3cb43a01e99e/image.png
﻿
LSTM-MSNet can be divided into 4 primary parts:
﻿
1) **Time Series Pre-processing**:
 
 - Normalization : For normalizing the Time Series Data, the mean of a time series serves as the scaling factor in the mean-scale transformation approach that the author suggests using. ﻿

This scaling strategy can be deﬁned as follows:
﻿https://storage.googleapis.com/slite-api-files-production/files/b4f4f758-4ce2-4310-a526-82f5dc63df03/image.png
Here, xi ,normalized represents the normalized observation, and k represents the number of observations of time series i.
 
 - Variance Stabilization Layer : After normalizing the time series, they stabilize the variance in the collection of time series by converting each time series to a logarithmic scale. The transformation can be defined in the following way: 
https://storage.googleapis.com/slite-api-files-production/files/975e4c7c-7f68-4f88-b6ca-687e79725a15/image.png
 
 - Moving Window Transformation : As a preprocessing step, they transform the past observations of time series (Xi) into multiple pairs of input and output frames using a Moving Window (MW) strategy. To summarize, the MW approach creates (K − n − m) recordings from a time series Xi of length K, with (m + n) observations in each record. In this case, n is the length of the input window (Past Period) and m is the length of the output window (Forecast Period). These frames are generated according to the Multi-Input Multi-Output (MIMO) principle used in multi-step forecasting, which directly predicts all the future observations up to the intended forecasting horizon.
https://storage.googleapis.com/slite-api-files-production/files/c1abbb30-7f07-49bc-b96e-0ea85b4a02de/image.png
The input window or the Past Period = n* output window or Forecast Period, with n being 1.5 in the paper. A very good example for understanding MIMO is:
https://storage.googleapis.com/slite-api-files-production/files/751ff70c-9209-46f4-8274-87774c4ccc56/image.png
﻿
2)  **Seasonal Decomposition** : 
When modelling seasonal time series with NNs, many studies suggest applying a prior seasonal adjustment, i.e., de-seasonalization to the time series. The main intention of this approach is to minimize the complexity of the original time series by  detaching the multi-seasonal components from a time series, and thereby reducing the subsequent effort of the NN’s learning process. Here, Multi-seasonal components refer to the repeating patterns that exist in a time series and that may change slowly over time.

The Author Proposes 5 Methods for Seasonal Decomposition : 
- Multiple STL Decomposition (MSTL)
- Seasonal-Trend decomposition by Regression (STR)
- Trigonometric, Box-Cox, ARMA, Trend, Seasonal (TBATS)
- Prophet
- Fourier Transformation
It is required by the applicant to implement any one of the above mentioned techniques.
https://storage.googleapis.com/slite-api-files-production/files/57a7a600-16fa-4444-977d-9398112b48b9/image.png
﻿
﻿
3) **Training Paradigms**:
The Author proposed 2 methods for training the LSTM model:
- Deseasonalised Approach (DS): This approach uses seasonally adjusted time series as moving window patches to train the LSTM-MSNet. Since the seasonal components are not included in DS for the training procedure, a reseasonalisation technique is later introduced in the Post-processing layer of LSTM-MSNet to ascertain the corresponding multiple seasonal components of the time series.
- Seasonal Exogenous Approach (SE): This second approach uses the output of the pre-processing layer, together with the seasonal components extracted from the multi-seasonal decomposition as external variables. Here, in addition to the normalized time series (without the deseasonalisation phase), the seasonal components relevant to the last observation of the input window are used as exogenous variables in each input window. As the original components of the time series are used in the training phase of SE, the LSTM-MSNet is expected to forecast all the components of a time series, including the relevant multi-seasonal patterns. Therefore, a reseasonalisation stage is not required by SE.

In summary, DS supplements the LSTM-MSNet by excluding the seasonal factors in the LSTM-MSNet training procedure. This essentially minimises the overall training complexity of the LSTM-MSNet. In contrast, SE supplements LSTM-MSNet in the form of exogenous variables that assist modelling the seasonal trajectories of a time series.
﻿
Fortunately, it is required by the applicant to implement Deseasonalised Approach (DS). Feel free to also implement Seasonal Exogenous Approach (SE), though it is not a mandatory criteria.
﻿
﻿
https://storage.googleapis.com/slite-api-files-production/files/c2cae6d9-0812-4c02-91bf-bab82a964fc8/image.png
﻿

LSTM Learning Scheme:
As highlighted earlier, the author uses the past observations of time series Xi, in the form of input and output windows to train the LSTM-MSNet. The author uses the LSTM model mentioned in this paper: https://arxiv.org/pdf/1909.00590.pdf, feel free to use any LSTM implementation as long as it is working in the right way.
﻿
https://storage.googleapis.com/slite-api-files-production/files/de07b98f-10c3-4dfb-9bd0-a3644ed4d501/image.png
﻿

Loss Function: 
The author uses the L1-norm, as the primary learning objective function, which essentially minimizes the absolute differences between the target values and the estimated values. They also include an L2-regularization term to minimize possible over fitting of the network
﻿
https://storage.googleapis.com/slite-api-files-production/files/4d90dec6-5e72-4d94-96ee-db4789164b81/image.png

4) **Post-processing Layer**:
The reseasonalisation and renormalisation is the main component of the post processing layer in LSTM-MSNet. Here, in the reseasonalisation stage, the relevant seasonal components of the time series are added to the forecasts generated by the LSTM. This is computed by repeating the last seasonal components of the time series to the intended forecast horizon. Next, in the renormalisation phase, the generated forecasts are back-transformed to their original scale by adding back the corresponding local normalization factor, and taking the exponent of the values. The final forecasts are obtained by multiplying this vector by the scaling factor used for the normalization process. 
﻿
## Code Contributions
This implementation extends the original LSTM-MSNet code with the following improvements and adaptations:
1	Dataset Adaptation: Adapted the code to handle a quarterly dataset instead of the hourly dataset used in the original paper. This required changes to the preprocessing pipeline and sequence creation logic.
2	Early Stopping Implementation: Introduced early stopping logic to automatically terminate training when validation loss stops improving, optimizing training time and preventing overfitting.
3	Optimized Hyperparameters: Adjusted training settings, reducing batch size (from 412 to 128) and epochs (from 500 to 100), ensuring efficient use of computational resources.
4	Refined create_sequences Function:
o	Ensured data consistency between input and target sequences.
o	Reformatted data for compatibility with LSTM models, addressing issues with mismatched dimensions in the original implementation.
5	Subset Training: Processed a 1,000-row subset from a 24,000-row quarterly dataset due to resource limitations, while ensuring it preserved key patterns and trends.
