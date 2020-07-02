# The new data set

We here provide two different data sets, generated by atomistic molecular dynamics (MD) simulation. In MD simulations, the chronological evolution of an N-particle system is computed by solving the Newton's equations of motion. Methodological developments in MD has pushed the limits of computable system-sizes to hundreds of millions of interacting particles, and timescales from femtoseconds (10<sup>-15</sup> second) to microseconds (10<sup>-6</sup> second), allowing all-atom simulations of an entire cell organelle. 

We have generated two data sets from two distinct kinds of MD simulation systems. The first data set is an equilibrium simulation of the enzyme adenosine kinase (ADK). The second one is a steered molecular dynamics (SMD) or non-equilibrium simulation of the 100-alanine polypeptide helix. In SMD, an external force is applied to the system along a chosen direction. We applied a force of 1 nanoNewton along one end of the 100-alanine helix, unfolding the protein.

We have generated high- as well as low-dimensional data for both the systems. In high-dimension, the position of every atom is explicitly defined, resulting in 3324 x 3 (for ADK) and 1003 x 3 (for 100-alanine) dimensions. For the low-dimensional data, positions of only the alpha-carbon atoms of each protein are defined, reducing the dimensionality of the problem to 214 x 3 and 100 x 3 respectively. For simplicity, we only provide the low dimensional data sets.

The data is in {X,Y,Z} format presenting the Cartesean coordinates of the atoms for every time point along the time series. A total of 10000 time points is considered for the ADK example distributed evenly across 10<sup>5</sup> (saved in steps of 10 fs), and similarly 2002 data points were generated for the 100-alanine example across 10<sup>7</sup> fs (saved in steps of 5000 fs. 
The equilibrium time series was simulated employing OpenMM, while the non-equilibrium data set was constructed using our NAMD molecular dynamics simulation software packages.

Raw trajectory files can be downloaded from the following google drive link: https://drive.google.com/drive/folders/1gxx-LV-UcQjBOgv7EjUaacQPXLGO0H1l?usp=sharing


# Generation of data set

## Equiliribum MD

A starting 3D protein model of ADK was generated using an x-ray diffraction crystal structure obtained from the protein data bank (PDB), available at https://www.rcsb.org . The atomic coordinates of ADK are encoded in the traditional PDB format presenting the {X, Y, Z} positions. X-ray is unable to resolve hydrogen atom positions. Thus, the position of hydrogen atoms were estimated before performing the simulation. The Amber force field, FF14SBonlysc, was used for this simulation. Addition of hydrogen atoms and the simulation were both performed using the OpenMM software package.
    
Script for running equilibrium simulations (which includes hydrogen addition) can be found in the Equilibrium_MD_simulation folder, alond with the PDB file after addition of hydrogens. The script can be executed as:

`python run_ADK.py > run_ADK.log`
    
### Pre-requisites

* OpenMM: GPU install
We used the GPU-implementation of OpenMM, which can be installed from Anaconda cloud using the following command.
`conda install -c omnia/label/cuda92 -c conda-forge openmm`

* cuda 9.2
We used CUDA 9.2. However, OpenMM is available for other CUDA versions as well. Please refer to http://openmm.org for installation instructions as well as OpenMM tutorials.

    
## Non-equilibrium MD

The PDB file for 100-alanine helix is provided. CHARMM36m force field was used for this simulation, and the simulation was performed using the NAMD software package. A simulation time  of 10<sup>7</sup> fs was required for extension of the helix to random coil.

Scripts for running SMD simulations has been provided in the Non-equilibrium_MD_simulation folder, along with the PDB of 100-alanine. The script can be executed as:

`namd2 smd.constvel.namd > smd.constvel.log`   

### Pre-requisites

* NAMD versions can be downloaded from https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=NAMD. 
* Details of the SMD simulation can be obtained in a tutorial form from http://www.ks.uiuc.edu/Training/Tutorials/science/10Ala-tutorial/tutorial-html/index.html
* PDF of the tutorial is available at http://www.ks.uiuc.edu/Training/Tutorials/science/10Ala-tutorial/10Ala-tutorial.pdf). 
* More information on NAMD is available at https://www.ks.uiuc.edu/Research/namd/.
    

# Data visualization

As mentioned above, the data presented here are the Cartesean coordinates of atoms for every time point in the time series. The VMD software package can be used to visualize the atoms in 3D space as well as observing the evolution of the atom positions in time, in the form of a molecular movie. VMD can be launched from the command line by typing 'vmd', and the menu options can be used to load the *.xyz file. Conversely, the *.xyz file can be loaded during launch with the following command:

`vmd filename.xyz`

A movie demonstrating the process of launching VMD and visualizing the reduced dimensonality data for 100-alanine has been uploaded in the google drive.

### Pre-requisites

* VMD versions can be downloaded from https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD
* More information on VMD is available at https://www.ks.uiuc.edu/Research/vmd/


# Data loading and analysis

## Loading data

The data can be loaded as a 3-dimensional numpy array using the script load_data.py, as demonstrated in the script load.py. Both the scripts are provided in the folers Equilibrium_MD_data and Non-equilibrium_MD_data. The script load_data.py returns:

* A 3D numpy array of dimensions T x N x 3, where T is the number of time points and N is the number of atoms
* N, which is the number of atoms (second dimension size of the 3D array).

## Data analysis

Most of our analysis on this data is currently not published. However, we have provided the script that we have used to generate the preliminary results, in the Analysis folder. The script can be executed as:

`python LSTM.py <lead_time> <n_hidden> <training_iters> <history> > LSTM.log`

Note that the name of the data file needs to be manually set in the code, at line 132.