#!date
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier
import torch
import RNN_custom

USE_CUDA = False  # Set this to False if you don't want to use CUDA
NUM_CV_STEPS = 10  # Number of randomized search steps to perform

n_samples = 100
n_features = 10
n_timepoints = 50
n_labels = 30
X = torch.rand((n_samples, n_timepoints, n_features))
y = torch.randint(0, n_labels, (n_samples,))
print(X.shape, y.shape)


steps = [('net', NeuralNetClassifier(
        RNN_custom.RNNClassifier,
        device=('cuda' if USE_CUDA else 'cpu'),
        max_epochs=5,
        lr=0.01,
        optimizer=torch.optim.RMSprop,
    ))
]


pipe = Pipeline(steps)
params = {
    'net__module__rec_layer_type': ['gru', 'lstm'],
    'net__module__num_units': [1, 2, 4, 8, 16 , 32],
    'net__module__num_layers': [1, 2, 3],
    'net__module__dropout': stats.uniform(0, 0.9),
    'net__lr': [10**(-stats.uniform(1, 5).rvs()) for _ in range(NUM_CV_STEPS)],
    'net__max_epochs': [5, 10],
}

clf = RandomizedSearchCV(pipe, params, n_iter=NUM_CV_STEPS, verbose=2, refit=False, scoring='accuracy', cv=3)
search = clf.fit(X, y)
print(search)
print(search.best_score_, search.best_params_)
