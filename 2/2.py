# Necessary imports
import matplotlib.pyplot as plt
import numpy as np

from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator

from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.neural_networks import EffectiveDimension
from qiskit_machine_learning.neural_networks import SamplerQNN

from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation


# 관측 데이터
# plotsX = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8]
# plotsY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# plotsZ = [np.float64(0.6649308002509239), np.float64(0.5432469780233532), np.float64(0.5229216536128067), np.float64(0.4136303303908814), np.float64(0.34040073893108064), np.float64(0.3334674067942181), np.float64(0.29828272312517734), np.float64(0.2931613106035447), np.float64(0.2617195887754421), np.float64(0.8765115115999842), np.float64(0.7086458511193956), np.float64(0.5563242705256354), np.float64(0.49427791173992724), np.float64(0.45726398349711467), np.float64(0.4093308963831607), np.float64(0.3572089521417598), np.float64(0.3432668824009466), np.float64(0.32126345114734706), np.float64(0.8480458798910582), np.float64(0.7833770184407413), np.float64(0.6432648893956592), np.float64(0.6017693562148643), np.float64(0.5141647732688566), np.float64(0.45528344354373845), np.float64(0.3845994405395859), np.float64(0.36099737464955584), np.float64(0.34608964334748454), np.float64(0.8066102969092795), np.float64(0.7314759144099704), np.float64(0.5514357424698489), np.float64(0.5056248843003626), np.float64(0.4415147563014448), np.float64(0.3735222644711333), np.float64(0.33095173259666616), np.float64(0.31145847476944993), np.float64(0.28465586965051226), np.float64(0.7291919955365915), np.float64(0.6152694737465726), np.float64(0.4988852631781797), np.float64(0.41935444799111155), np.float64(0.3622664134270944), np.float64(0.31847893811367356), np.float64(0.2798511935861135), np.float64(0.2588215306839434), np.float64(0.23489069953356467), np.float64(0.650672239418716), np.float64(0.5445354716901776), np.float64(0.42195440828435055), np.float64(0.35618573263365166), np.float64(0.3174190831422259), np.float64(0.27310150717016296), np.float64(0.2422928601376879), np.float64(0.22108413309342437), np.float64(0.20294336705124494), np.float64(0.6375806961546531), np.float64(0.4804283191039677), np.float64(0.3811329777803839), np.float64(0.32299101003615277), np.float64(0.2787662894810898), np.float64(0.24040494169015028), np.float64(0.21381606994773963), np.float64(0.19255164387542706), np.float64(0.17734769491504598), np.float64(0.5724096493022294), np.float64(0.4352944394906318), np.float64(0.3374567239757281), np.float64(0.28929330608539483), np.float64(0.24517657969202308), np.float64(0.21427105203385083), np.float64(0.19047793302652744), np.float64(0.17273186566519827), np.float64(0.15778998761073856)]

plotsX = []
plotsY = []
plotsZ = []

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# set random seed
algorithm_globals.random_seed = 42
sampler = Sampler()
estimator = Estimator()

# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1") % 2

for num_qubits in [1, 2, 3, 4, 5, 6, 7, 8]:
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
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
        plotsY.append(depth)
        plotsZ.append(global_eff_dim_1 / d)

# plot the normalized effective dimension for the model
ax.scatter(plotsX, plotsY, plotsZ, c=plotsY, cmap='viridis', label='Model Points')
# ax.plot_trisurf(plotsX, plotsY, plotsZ, cmap='viridis', label='Model Points')
ax.set_xlabel("Number of qubits")
ax.set_ylabel("Depth of the circuit")
ax.set_zlabel("Normalized GLOBAL effective dimension")
ax.legend()
# ax.view_init(elev=0, azim=0)
# ax.view_init(elev=0, azim=-90)
plt.show()

'''
def update(frame):
    ax.view_init(elev=30, azim=frame)
    return fig,

ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=40)

print("Saving GIF file... (This may take a moment)")
ani.save('rotating_3d_plot.gif', writer='pillow', fps=25)
print("Done!")
'''
