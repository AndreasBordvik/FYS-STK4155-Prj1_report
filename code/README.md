# Setting up the environment
We use a conda environment to install the dependencies for this assignment.
To set up the environment, follow these steps:
1. Make sure conda (by anaconda or miniconda) is installed.
2. In the assignment directory, create and activate the conda environment for the assignment by running:
   ```sh
   conda env create -f environment.yml
   conda activate fys-stk-1
   ```
3. To add the conda environment to Jupyter, run (still in the fys-stk-1 environment):
   ```sh
   python -m ipykernel install --user --name=fys-stk-1
   ```
4. Ever from the same environment, run
   ```sh
   jupyter-lab
   ```
   to start the Jupyter server. From here, the notebooks for each of the questions can be opened and run.

5. Notes for running the notebooks:
   You may need to have latex locally installed in order to run using the fonts defined together with 
   the import statements. 
   ```sh
   plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 10,
	})
   ```