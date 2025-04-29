This is a large project that creates and analyzes Molecular Dynamics(MD) simulations using LAMMPS (https://www.lammps.org/). The primary system of study is the passivating interface of a TOPCon solar cell, i.e., a c-Si/a-SiOx stack, but these tools can be used on any similar system. This project includes lots of code for creating, testing, and analyzing these samples, but the bulk of the code is a large Nudged Elastic Band (https://doi.org/10.1063/1.1323224) data pipeline to study the diffusion of atoms in these samples. NEB is a method that can calculate minimum energy paths for a reaction of interest. In this project, the reaction is an atom or set of atoms moving from one location to another, which allows us to develop an atomic-level understanding of the diffusion of atoms or complexes in the material. 

# NEB Pipeline
<img src="https://github.com/user-attachments/assets/b19c814b-41b7-4d7a-ba66-2313e4d91797" align="right" width="30%" alt="Example diffusion process whose energy barrier has been calculated using this NEB pipeline">

The NEB method requires the initial and final states of the reaction, so this pipeline speeds up the procedure of creating these two data files so that more NEB calculations can be done. An example diffusion process whose energy barrier has been calculated using this NEB pipeline is shown on the right. Significant understanding of the initial and final states is needed to be able to develop the code that creates the correct data files.

The important steps and files in the NEB pipeline;

1) Create a sample and minimize it to the proper energy tolerance of the pipeline(crucial for speeding up the pipeline). There are several example files in the lmp/ folder that create and minimize samples.
2) Create the 'pair list' that is passed into the pipeline. The pair list describes which atoms are being tested and what their final locations are. This file can be manually created or by using an algorithm like those found in [py/CreatePairList.py](https://github.com/zimanyi-group/TOPCON-MD/blob/2130c84a616471efe19783c0e83591f3746cddbb/py/CreatePairList.py#L289) to go through all the atoms in your sample to test which ones fit the criteria you would like to test.
3) Run the pipeline on your pair list. This is done using a bash file which can be run locally [run-NEB.sh](run-NEB.sh) or on a HPC cluster [farm-run-neb.sh](farm-run-neb.sh). These scripts loop through the given pair list and run;
    - A python script to set up the initial and final NEB data files - [PrepNEB.py](py/PrepNEB.py),
     - A LAMMPS neb procedure - [NEB.lmp](lmp/NEB.lmp), more information can be found [here](https://docs.lammps.org/neb.html) and finally
     - A python script to analyze the results, save the data, and create a plot/animation of the process which the NEB calculation was done on - [Process-NEB.py](py/Process-NEB.py).
4) The polished images/gifs and csv file containing the pipeline results default to [neb-out/](/neb-out/) while the specific log and debug files are placed in the [output/](output/) folder for deeper analysis or bug fixing.


# System requirements
The pipeline requires python with a number of common libraries as well as the Ovito python library, which can cause conflicts with other plotting libraries. A working conda environment with strict versions of each of the necessary libraries is included in [current_conda_env.yml](current_conda_env.yml). You may need to further install matplotlib via pip and not conda, but I was able to manage the entire environment within conda on my home PC.

An installation of LAMMPS which was built in shared library mode with python support as well as the replica package. An example build script using cmake is provided in [buildlmp.sh](/buildlmp.sh).


