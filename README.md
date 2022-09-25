# ti-DEMPM 
                                                                                                                - developed by Shi-Yihao Zhejiang Universiy
A high performance objected-oriented Discrete Element Method- Material Point Method (DEM-MPM) simulator in [Taichi](https://github.com/taichi-dev/taichi). (In Progress)

## Examples

## Features
The features are listed as follows:
1. Discrete Element Method (only spherical particles supported)
2. Material Point Method (Linear elastic / Von-Mises / Tresca / Mohr-Coumlob / Drucker-Parger / Nowtonian Fluid / Non-Newtonian Fluid / Elas-plastic)
3. DEMPM coupling (Penlaty Method / Incremental potential contact)

## Future Work
1. Multisphere particles (clump) are supported.
2. GPU memory allocate
3. More constitutive models in MPM
4. Pre-progress (i.e., input obj files and etc.)

## Install
1. Install essential dependencies
```
bash requirements.sh
```
2. Set up eniveronmental variables
```
sudo gedit ~/.bashrc
```
Add the installation path to bashrc file:
```
export tiDEMPM=~/tiDEMPM
```

## Acknowledgememt
Implementation is largely inspired by [ComFluSoM](https://github.com/peizhang-cn/ComFluSoM).
