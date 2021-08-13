# Temperature Scaling
A simple way to calibrate your neural network.
The `temperature_scaling.py` module can be easily used to calibrated any trained model.

Based on results from [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

## Motivation

**TLDR:** Neural networks tend to output overconfident probabilities.
Temperature scaling is a post-processing method that fixes it.

**Long:**

Neural networks output "confidence" scores along with predictions in classification.
Ideally, these confidence scores should match the true correctness likelihood.
For example, if we assign 80% confidence to 100 predictions, then we'd expect that 80% of
the predictions are actually correct. If this is the case, we say the network is **calibrated**.

A simple way to visualize calibration is plotting accuracy as a function of confidence.
Since confidence should reflect accuracy, we'd like for the plot to be an identity function.
If accuracy falls below the main diagonal, then our network is overconfident.
This happens to be the case for most neural networks, such as this ResNet trained on CIFAR100.

Temperature scaling is a post-processing technique to make neural networks calibrated.
After temperature scaling, you can trust the probabilities output by a neural network:

![Diagram](plots/rel_diagram_test.png)

![Diagram2](plots/conf_histogram_test.png)

Temperature scaling divides the logits (inputs to the softmax function) by a learned scalar parameter. I.e.
```
softmax = e^(z/T) / sum_i e^(z_i/T)
```
where `z` is the logit, and `T` is the learned parameter.
We learn this parameter on a validation set, where T is chosen to minimize NLL.

## References

### Metrics

ECE and MCE - [Obtaining Well Calibrated Probabilities Using Bayesian Binning](http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf)

SCE, ACE and TACE - [Measuring Calibration in Deep Learning](https://arxiv.org/abs/1904.01685)

### Recalibration Methods

Tempurature Scaling - [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)

### Visualizations

Reliability Diagram and Confidence Histograms - [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
