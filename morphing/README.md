# Registration with [ANTsPyX](https://pypi.org/project/antspyx/)

## Setting up your computer
The `ANTsPy` library is a wrapper for the C++ imaging processing [ANTs](https://github.com/ANTsX/ANTs) library.

Unfortunately, this library is only available for Linux and MacOS operative systems. Therefore, if you are working on a Windows computer, the first thing that needs to be done is install a Windows Subsystem for Linux or WSL.

### Installing WSL
You can easily install WSL following [this guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10). WSL1 is enough, and you don't need to do the extra steps to set up WSL2.

If you have problems installing a Linux distribution from the Windows Store, follow the next steps:
1. Download the distro `.appx` file.
2. Change it to a `.zip` file and extract it.
3. Run the `.exe` file in the folder.

This will install the distro; it won't appear on the Windows start menu but can be run from the same `.exe` file. However, the location of the folder cannot be changed after installation (otherwise it would not launch).

### Setting up your python ecosystem
Once this is done, you can access the Linux command prompt and start setting your system. First, install conda using:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
and 
```
bash Miniconda3-latest-Linux-x86_64.sh
```
Follow the instructions until the installation process is complete.


To continue, it is recommended to install the `rplab` environment by using the `environment.yml` file you can find in the [lab-docs](https://github.com/portugueslab/lab-docs) repository in GitHub. To do so, first clone the repository 
```
git clone https://github.com/portugueslab/lab-docs.git
```
and then install the environment.
```
conda env create -f {path to the envinroment.yml file}
```
You will additionally need to install ANTsPy in your rplab environment withing the Ubuntu system. Do that via:
```
pip install antspyx
```
Some additional packages may need to be installed as well.


### Running JupyterLab
You can now run jupyter notebook/lab from WSL. Just activate the rplab environment, run the usual command on the Linux terminal, and paste on your browser one of the URLs that the consol will return. It is recommended to add the following arguments:
```
jupyter lab --port=8889 --no-browser
```


## Morphing pipeline
The whole pipeline consists of two notebooks, one to be run in Windows and including some preprocessing of the stacks as well as an initial registration, and another one to be run in your Linux subsystem, performing the actual ANTsPy registration.
1. Run the Windows notebook and store its outputs.
2. Open JupyterLab in your Linux subsystem, and upload the data you saved in step 1.
3. Run the Linux notebook in JupyterLab with the data you just uploaded as an input.
4. Download to your Windows system the output of the second notebook.
