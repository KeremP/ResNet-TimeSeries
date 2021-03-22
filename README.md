# ResNet-TimeSeries
Pytorch implementation of Resnet for time-series prediction and use in Numerai tournament.

Based on: https://www.kaggle.com/a763337092/pytorch-resnet-starter-training

Several users on Numerai forums have found success using architectures with residual connections. This implementation isbased off an architecture used in Janestreet Market Prediction competition hosted on Kaggle. While not the perfect implementation of a ResNet, this implementation works well with multivariate time series data.

ToDo:
- Feature-Target neutralization
- Feed Eras as mini-batches during training
