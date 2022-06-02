# imports
import numpy as np
import argparse

from openmmtools.testsystems import AlanineDipeptideVacuum, AlanineDipeptideImplicit

from reform import simu_utils


# Parse input arguments
parser = argparse.ArgumentParser(description='Sample configurations of '
                                             'Alanine dipeptide with REMD')

parser.add_argument('--out_dir', type=str, default='./',
                    help='Directory where to save output')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed to be used for sampling')
parser.add_argument('--equil_iter', type=int, default=1000,
                    help='Number of iterations to run for equilibration')
parser.add_argument('--num_iter', type=int, default=10000,
                    help='Number of iterations to run the simulation')
parser.add_argument('--rec_iter', type=int, default=1000,
                    help='Recording interval')
parser.add_argument('--min_temp', type=float, default=300.,
                    help='Minimum temperature')
parser.add_argument('--max_temp', type=float, default=1300.,
                    help='Maximum temperature')
parser.add_argument('--step_temp', type=float, default=50.,
                    help='Temperature increment between replicas')
parser.add_argument('--env', type=str, default='vacuum',
                    help='Environment of the molecule, can be vacuum '
                         'or implicit for implicit solvent')

args = parser.parse_args()


# Simulation setup
interface = "single_threaded"

seed = args.seed
np.random.seed(seed)

TIME_STEP = 1   # in fs
SIMU_TIME = args.num_iter // 1000  # in ps
RECORDING_INTERVAL = args.rec_iter // 1000   # in ps
EXCHANGE_INTERVAL = 0.2  # in ps; or 0. when you don't need the exchanges

temps_intended = list(np.arange(args.min_temp, args.max_temp + 1e-3,
                                args.step_temp))

OUTPUT_PATH = args.out_dir + 'aldp_%010i.npy' % seed

# Prepare the MultiTSimulation
n_replicas = len(temps_intended)

# Setup Alanine dipeptide
if args.env == 'vacuum':
    system = AlanineDipeptideVacuum(constraints=None)
elif args.env == 'implicit':
    system = AlanineDipeptideImplicit(constraints=None)
else:
    raise NotImplementedError('This environment is not implemented.')

integrator_params = {"integrator": "Langevin",
                     "friction_in_inv_ps": 1.0,
                     "time_step_in_fs": 1.0}

simu = simu_utils.MultiTSimulation(system.system, temps_intended, interface=interface,
                                   integrator_params=integrator_params, verbose=False,
                                   platform='Reference')

# Set seed of integrators
for i in range(n_replicas):
    simu._context._integrators[i].setRandomNumberSeed(seed)

simu.set_positions([system.positions] * n_replicas)
simu.minimize_energy()
simu.set_velocities_to_temp()
simu.run(args.equil_iter)  # pre-equilibration
_ = simu.reset_step_counter()

# Install the simulation hooks for recording and exchange
simu_steps = simu_utils.recording_hook_setup(simu=simu, simu_time=SIMU_TIME,
                                             recording_interval=RECORDING_INTERVAL,
                                             output_path=OUTPUT_PATH,
                                             exchange_interval=EXCHANGE_INTERVAL)

simu.run(simu_steps)