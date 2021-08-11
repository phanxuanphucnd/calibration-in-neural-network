# Temperature Scaling

A simple way to calibrate your neural network.
The `temperature_scaling.py` module can be easily used to calibrated any trained model.

Based on results from [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).

- Neural networks tend to output overconfident probabilities. Temperature scaling is a post-processing method that fixes it. 
- Calibrated confidence estimates are also important for model interpretability.
- For example, if we assign 80% confidence to 100 predictions, then we'd expect that 80% of the predictions are actually correct. If this is the case, we say the network is calibrated.

A simple way to visualize calibration is plotting accuracy as a function of confidence. Since confidence should reflect accuracy, we'd like for the plot to be an identity function. If accuracy falls below the main diagonal, then our network is overconfident. This happens to be the case for most neural networks.

![Uncalibrated ResNet](https://user-images.githubusercontent.com/824157/28974416-51ba7be4-7904-11e7-89ff-3c9b0ec4b607.png)

Temperature scaling is a post-processing technique to make neural networks calibrated.
After temperature scaling, you can trust the probabilities output by a neural network:

![Calibrated ResNet](https://user-images.githubusercontent.com/824157/28974415-51ae78a8-7904-11e7-9b33-8fbe1f7c0a53.png)

#### Temperature scaling

Temperature scaling divides the logits (inputs to the softmax function) by a learned scalar parameter. I.e.
```
softmax = e^(z/T) / sum_i e^(z_i/T)
```
where `z` is the logit, and `T` is the learned parameter. We learn this parameter on a validation set, where T is chosen to minimize NLL.

#### Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{m=1}^{M} \frac{\lvert B_m \rvert}{n} \big\lvert \text{acc}(B_m) - \text{conf}(B_m) \big\rvert$$

where $B_m$ is bin. (e.g. a 80%-85% bin). We take a weighted average of the bins' $\vert \text{acc} - \text{conf} \vert$ difference. 

- The difference between acc and conf for a given bin represents the calibration gap (red bars in reliability diagrams)

- Although increasing depth and width may reduce classification error, we observe that these increases negatively affect model calibration. 

-  NNs can overfit to NLL without overfitting to the 0/1 loss.
- Surprisingly, overfitting to NLL is beneficial to classification accuracy. On CIFAR-100, test error drops from 29% to 27% in the region where NLL overfits. This phenomenon renders a concrete explanation of miscalibration: the network learns better classification accuracy at the expense of well-modeled probabilities.
- There are two cases where NLL increases,
    1. Prediction is correct, and the model tries to reduce max softmax value. (and probably change the prediction)
    2. Prediction is wrong, and the model tries to increase max softmax value (output 0.95 even if the prediction is wrong)
- As seen in the graph, we can guess 2 is the case. (because accuracy didn't get worse.) 

#### Calibration Methods

- Platt scaling (Platt et al., 1999) : $ \hat{q} = \sigma (az + b) $
- Temperature scaling : $ \text{softmax}(\frac{\text{logits}}{\text{T}})$
    - $T$ is called the temperature, and it “softens” the softmax (i.e. raises the output entropy) with $T$ > 1.

- $T$ is optimized with respect to NLL on the validation set. Temperature scaling does not affect the model’s accuracy because the parameter $T$ does not change the maximum of the softmax function.


## Demo

First train a DenseNet on CIFAR100, and save the validation indices:
```sh
python train.py --data <path_to_data> --save <save_folder_dest>
```

Then temperature scale it
```sh
python demo.py --data <path_to_data> --save <save_folder_dest>
```


## To use in a project

Copy the file `temperature_scaling.py` to your repo.
Train a model, and **save the validation set**.
(You must use the same validation set for training as for temperature scaling).
You can do something like this:

```python
from temperature_scaling import ModelWithTemperature

orig_model = ... # create an uncalibrated model somehow
valid_loader = ... # Create a DataLoader from the SAME VALIDATION SET used to train orig_model

scaled_model = ModelWithTemperature(orig_model)
scaled_model.set_temperature(valid_loader)
```
