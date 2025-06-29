# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator

from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EffectiveDimension, LocalEffectiveDimension
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from mpl_toolkits.mplot3d import Axes3D

# set random seed
algorithm_globals.random_seed = 42
sampler = Sampler()
estimator = Estimator()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2

plotsX = []
plotsY = []
plotsZ = []

for num_qubits in range(3,10):
    for depth in range(1, 10):
        # combine a custom feature map and ansatz into a single circuit
        qc = QNNCircuit(
            feature_map=ZFeatureMap(feature_dimension=num_qubits, reps=depth),
            ansatz=RealAmplitudes(num_qubits, reps=depth),
        )
        
        output_shape = 2  # corresponds to the number of classes, possible outcomes of the (parity) mapping.

        # construct QNN
        qnn = SamplerQNN(
            circuit=qc,
            interpret=parity,
            output_shape=output_shape,
            sparse=False,
            sampler=sampler,
        )

        # we can set the total number of input samples and weight samples for random selection
        num_input_samples = 10
        num_weight_samples = 10

        global_ed = EffectiveDimension(
            qnn=qnn, weight_samples=num_weight_samples, input_samples=num_input_samples
        )

        # we can also provide user-defined samples and parameters
        input_samples = algorithm_globals.random.normal(0, 1, size=(10, qnn.num_inputs))
        weight_samples = algorithm_globals.random.uniform(0, 1, size=(10, qnn.num_weights))

        global_ed = EffectiveDimension(qnn=qnn, weight_samples=weight_samples, input_samples=input_samples)

        # finally, we will define ranges to test different numbers of data, n
        n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

        global_eff_dim_1 = global_ed.get_effective_dimension(dataset_size=n[6])

        d = qnn.num_weights    

        print("Effective dimension: {}".format(global_eff_dim_1))
        print("Number of weights: {}".format(d))

        plotsX.append(num_qubits)
        plotsY.append(global_eff_dim_1 / d)
        plotsZ.append(depth)

# plot the normalized effective dimension for the model
ax.scatter(plotsX, plotsY, plotsZ, c=plotsY, cmap='viridis', label='Model Points')
ax.set_xlabel("Number of qubits")
ax.set_ylabel("Normalized GLOBAL effective dimension")
ax.set_zlabel("Depth of the circuit")
ax.legend()
plt.show()