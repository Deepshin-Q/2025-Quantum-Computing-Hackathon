import os
from qiskit_ionq import IonQProvider
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

provider = IonQProvider(os.getenv("IONQ_API_KEY"))
simulater_backend = provider.get_backend("simulator")
simulater_backend.set_options(noise_model="aria-1")

qc = QuantumCircuit(2, 2)
qc.measure_all()

transpiled_qc = transpile(qc, backend=simulater_backend)

job = simulater_backend.run(transpiled_qc, shots=10000)
result = job.result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts)
plt.show()