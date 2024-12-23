# Steps to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Train model:
   ```bash
   python src/train.py
   ```

# To deploy to GitHub
1. Create a new GitHub repository
2. Initialize git in your local project:
   ```bash
   git init
   ```
3. Push your code to the new repository:
   ```bash
   git remote add origin https://github.com/your-username/your-repo.git
   git branch -M main
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

4. The GitHub Actions workflow will automatically trigger when you push to the repository. It will:
   - Set up the Python environment
   - Install dependencies
   - Run all tests to verify:
     - Model validity
     - Parameter count

# Models Comparison
Overall Target:
<ol>
<li>99.4% Accuracy</li>
<li>Less than or equal to 15 Epochs</li>
<li>Less than 8000 Parameters</li>
<li>Iterative approach building atleast 3 models </li>
</ol>
<table>
        <tr>
                <th>Target</th>
                <th>Result</th>
                <th>Analysis</th>
        </tr>
        <tr><td>Model_1.py</td></tr>
        <tr>
                <td>
                        <ol>
                        <li>Building Base model with base skelton</li>
                        <li>Using correct transforms, data sets and architecture </li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters: 29k</li>
                        <li>Best Training Accuracy: 99.69%</li>
                        <li>Best Test Accuracy: 99.05%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Large number of parameters</li>
                        <li>Gap between training and test accuracy (Over Fitting)</li>
                        </ol>
                </td>
        </tr>
        <tr><td>Model_2.py</td></tr>
        <tr>
                <td>
                        <ol>
                        <li>Lighter model with lesser parameters</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters: 14.5k</li>
                        <li>Best Training Accuracy: 99.54%</li>
                        <li>Best Test Accuracy: 98.80%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters are still more than 8k</li>
                        <li>Big Gap between training and test accuracy (Over Fitting)</li>
                        </ol>
                </td>
        </tr>
        <tr><td>Model_3.py</td></tr>
        <tr>
                <td>
                        <ol>
                        <li>Reduce the gap between training and test accuracy</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters: 14.5k</li>
                        <li>Best Training Accuracy: 99.56%</li>
                        <li>Best Test Accuracy: 99.11%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters are still more than 8k</li>
                        <li>Better gap between training and test accuracy than last one but still overfitting</li>
                        <li>Good model with slowly increasing accuracy each epoch with a decent gap between training and test accuracy</li>
                        </ol>
                </td>
        </tr>
        <tr><td>Model_4.py</td></tr>
        <tr>
                <td>
                        <ol>
                        <li>Reduce the number of parameters to less than 8k still maintaining the gap between training and test accuracy</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters: 7k</li>
                        <li>Best Training Accuracy: 97.73%</li>
                        <li>Best Test Accuracy: 97.39%</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters are less than 8k (Met one overall target)</li>
                        <li>Good model with slowly increasing accuracy each epoch with a decent gap between training and test accuracy</li>
                        <li>Can be trained further to get better results</li>
                        <li>Training accuracy has reduced but that may be due to reduction of total parameters</li>
                        </ol>
                </td>
        </tr>
        <tr><td>Model_5.py</td></tr>
        <tr>
                <td>
                        <ol>
                        <li>Increase the accuracy while still keeping the parameters under 8k.</li>
                        <li>Keeping the gap between training and test accuracy low</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters: 7,778</li>
                        <li>Best Training Accuracy: 98.75%</li>
                        <li>Best Test Accuracy: 99.44% (validation accuracy: 99.50%)</li>
                        </ol>
                </td>
                <td>
                        <ol>
                        <li>Parameters are less than 8k (Met overall target)</li>
                        <li>Good model with high accuracy and low gap between training and test accuracy</li>
                        <li>Met all the targets</li>
                        </ol>
                </td>
        </tr>
</table>

# Training Logs
Logs of training Model_5 (can also be found in src/playground.ipynb)
```
[INFO] Using device: mps
[STEP 1/5] Preparing datasets...
[INFO] Dataloader arguments: {'shuffle': True, 'batch_size': 128, 'num_workers': 4, 'pin_memory': True}
[INFO] Total training batches: 469
[INFO] Batch size: 128
[INFO] Training samples: 60000
[INFO] Test samples: 10000

[INFO] Validation samples: 1000

[STEP 2/5] Initializing model...
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
           Dropout-3            [-1, 8, 28, 28]               0
              ReLU-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 12, 28, 28]             876
       BatchNorm2d-6           [-1, 12, 28, 28]              24
           Dropout-7           [-1, 12, 28, 28]               0
              ReLU-8           [-1, 12, 28, 28]               0
         MaxPool2d-9           [-1, 12, 14, 14]               0
           Conv2d-10            [-1, 8, 16, 16]             104
           Conv2d-11            [-1, 8, 14, 14]             584
      BatchNorm2d-12            [-1, 8, 14, 14]              16
          Dropout-13            [-1, 8, 14, 14]               0
             ReLU-14            [-1, 8, 14, 14]               0
           Conv2d-15            [-1, 8, 12, 12]             584
      BatchNorm2d-16            [-1, 8, 12, 12]              16
          Dropout-17            [-1, 8, 12, 12]               0
             ReLU-18            [-1, 8, 12, 12]               0
           Conv2d-19           [-1, 12, 10, 10]             876
      BatchNorm2d-20           [-1, 12, 10, 10]              24
          Dropout-21           [-1, 12, 10, 10]               0
             ReLU-22           [-1, 12, 10, 10]               0
           Conv2d-23             [-1, 12, 8, 8]           1,308
      BatchNorm2d-24             [-1, 12, 8, 8]              24
          Dropout-25             [-1, 12, 8, 8]               0
             ReLU-26             [-1, 12, 8, 8]               0
           Conv2d-27             [-1, 12, 6, 6]           1,308
      BatchNorm2d-28             [-1, 12, 6, 6]              24
          Dropout-29             [-1, 12, 6, 6]               0
             ReLU-30             [-1, 12, 6, 6]               0
           Conv2d-31             [-1, 16, 4, 4]           1,744
AdaptiveAvgPool2d-32             [-1, 16, 1, 1]               0
           Conv2d-33             [-1, 10, 1, 1]             170
================================================================
Total params: 7,778
Trainable params: 7,778
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 0.03
Estimated Total Size (MB): 0.70
----------------------------------------------------------------
/Users/gitesh.grover/Study/AI-ERA/venv/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
[STEP 3/5] Starting training and Testing...

[INFO] Training of Epoch 1 started...
Epoch 1: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 80.10it/s, loss=0.301, accuracy=90.33%]
[INFO] Training of Epoch 1 completed in 5.86 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 97.34%

[INFO] Training of Epoch 2 started...
Epoch 2: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 89.89it/s, loss=0.109, accuracy=96.64%]
[INFO] Training of Epoch 2 completed in 12.88 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.18%

[INFO] Training of Epoch 3 started...
Epoch 3: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 86.79it/s, loss=0.093, accuracy=97.17%]
[INFO] Training of Epoch 3 completed in 19.99 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.43%

[INFO] Training of Epoch 4 started...
Epoch 4: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 88.29it/s, loss=0.082, accuracy=97.48%]
[INFO] Training of Epoch 4 completed in 27.03 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.91%

[INFO] Training of Epoch 5 started...
Epoch 5: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 87.80it/s, loss=0.074, accuracy=97.69%]
[INFO] Training of Epoch 5 completed in 34.10 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 99.01%

[INFO] Training of Epoch 6 started...
Epoch 6: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 88.42it/s, loss=0.069, accuracy=97.86%]
[INFO] Training of Epoch 6 completed in 41.12 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.89%

[INFO] Training of Epoch 7 started...
Epoch 7: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 87.88it/s, loss=0.062, accuracy=98.05%]
[INFO] Training of Epoch 7 completed in 48.19 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.94%

[INFO] Training of Epoch 8 started...
Epoch 8: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 87.07it/s, loss=0.061, accuracy=98.17%]
[INFO] Training of Epoch 8 completed in 55.31 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 99.18%

[INFO] Training of Epoch 9 started...
Epoch 9: 100%|████████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 87.40it/s, loss=0.059, accuracy=98.14%]
[INFO] Training of Epoch 9 completed in 62.48 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 99.20%

[INFO] Training of Epoch 10 started...
Epoch 10: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 86.94it/s, loss=0.060, accuracy=98.12%]
[INFO] Training of Epoch 10 completed in 69.66 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 98.82%

[INFO] Training of Epoch 11 started...
Epoch 11: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 86.61it/s, loss=0.045, accuracy=98.59%]
[INFO] Training of Epoch 11 completed in 76.83 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.38%

[INFO] Training of Epoch 12 started...
Epoch 12: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 85.99it/s, loss=0.042, accuracy=98.66%]
[INFO] Training of Epoch 12 completed in 84.14 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.37%

[INFO] Training of Epoch 13 started...
Epoch 13: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 87.32it/s, loss=0.039, accuracy=98.72%]
[INFO] Training of Epoch 13 completed in 91.30 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.38%

[INFO] Training of Epoch 14 started...
Epoch 14: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 88.69it/s, loss=0.040, accuracy=98.75%]
[INFO] Training of Epoch 14 completed in 98.36 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.44%

[INFO] Training of Epoch 15 started...
Epoch 15: 100%|███████████████████████████████████████████████████████████████████| 469/469 [00:05<00:00, 88.45it/s, loss=0.041, accuracy=98.75%]
[INFO] Training of Epoch 15 completed in 105.45 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.44%

[STEP 4/5] Evaluating model against validation...
Test Accuracy: 99.50%

![Graph Not Loaded](static/images/Accuracy-Graph.png?raw=true "Training and Testing Accuracy Graph")
```

