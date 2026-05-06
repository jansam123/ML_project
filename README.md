# Getting to Snellius

1. Look for an email from SURFcua with the subject "SURFcua new login scur0034 created". In this email, you will find your username. To get the password, you need to reset [here](https://sso.cua.surf.nl/realms/cua/login-actions/reset-credentials). Follow the instructions to reset your password.
2. Open Snellius On Demand at [https://ondemand.snellius.surf.nl/](https://ondemand.snellius.surf.nl/).
3. Click on VSCode.
4. Set `gpu_h100` partition. Change time to 12:00:00. Set number of CPU cores to 16. Set memory to 32GB. Request 1 GPU. Click on "Launch".
5. Wait for the VSCode session to start. Once it is ready, click on "Connect to VSCode".
6. Click on the three horizontal lines in top left on the VSCode interface. Click on "Terminal" and then "New Terminal". This will open a terminal in the bottom of the VSCode interface.
7. Type `git clone https://github.com/jansam123/ML_project.git` to clone the repository. Then, type `cd ML_project` to navigate to the project directory.
8. You can either work from the home directory and see all the files, or it's nicer to reopen the VSCode interface in the `ML_project` directory. To do this, click on the three horizontal lines in top left on the VSCode interface. Click on "File" and then "Open Folder". Navigate to the `ML_project` directory and click "Open". This will open the project in the VSCode interface, and you can see all the files in the left sidebar. The path should like `/home/scur0034/ML_project`, where `scur0034` is your username.
9. You can then click on README.md to see these instructions


# Setting up conda environment
1. To properly use the correct and latest packages, we will install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and create a conda environment. 
2. Run this commands to install conda:
   ```bash
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p ~/conda
   echo "source  ~/conda/bin/activate" >> ~/.bashrc
   source ~/.bashrc
   rm Miniforge3-Linux-x86_64.sh
   ```
3. Create a conda environment with latest python:
    ```bash 
    conda create -n ml_project 
    ```
4. Activate the conda environment:
    ```bash
    conda activate ml_project
    ```
5. Install pytorch:
    ```bash
    conda install pytorch
    ```
6. Check GPU is available:
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ``` 
# Getting the dataset.
The dataset is already downloaded on Snellius, and it's available at `FIX THIS`. 
However, if you want to download the dataset yourself, you can use the `get_datasets.py` script available in the repository. Be aware this can take some time to download.
```bash
    python get_datasets.py JetClass -d data
```