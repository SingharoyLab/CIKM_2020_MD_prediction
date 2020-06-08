from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

print('loading structure')
pdb = PDBFile('4ake_monomer.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('/home/eawilso6/apps/openmm-forcefields/amber/ffxml/protein.ff14SBonlysc.xml')
print('adding hydrogens')
modeller.addHydrogens(forcefield)
print('adding solvent')
print('minimizing')
system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolvent=GBn2)
integrator = LangevinIntegrator(310*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()
print('saving..')
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('4ake_monomer-solvated.pdb', 'w'))
print('Done')

print('starting simulation')
simulation.reporters.append(DCDReporter('4ake_output.dcd', 10))
simulation.reporters.append(StateDataReporter('log.txt', 10, progress=True, time=True, 
                                              temperature=True, totalSteps=1000000, separator=' '))
#simulation.reporters.append(StateDataReporter('run_log.txt', 1000, time=True, progress=True, step=True, potentialEnergy=True, temperature=True, separator='  '))
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 1000000))
simulation.step(1000000)
