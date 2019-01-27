# Portfolio-Optimization
It includes several popular portfolio optimization methods

Methods: Min Variance, Max Diversification, Risk Contribution Parity, Min CVaR, Inverse Volatility
Most of them involves compute the covariance matrix, so I include several covariance shrink method in sklearn --- 'LedoitWolf','MinDet'

In addition, you can add some penality to Min Variance, Max Diver, Min CVaR to force them to select more options.
Please see the code to understand more, different methods may need different input parameters.

You may find more information by reading my bachelor thesis 'Econ', but I don't polish it since I graduate, so don't expect too much
