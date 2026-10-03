import os
import sys

root = r'C:\Users\HariharanVS\Desktop\AI product review intelligence'
if root not in sys.path:
    sys.path.insert(0, root)

from src.train import train_and_evaluate_model

print('starting_train')
metrics = train_and_evaluate_model()
print('done_train')
print(metrics[1]['logistic_regression']['macro_f1'])
print(metrics[1]['linear_svm']['macro_f1'])
print('artifacts_exist', os.path.exists(os.path.join(root, 'models', 'logistic_regression.joblib')), os.path.exists(os.path.join(root, 'models', 'metrics.json')))
