# ti-DEMPM 
A high performance objected-oriented Discrete Element Method- Material Point Method (DEM-MPM) simulator in [Taichi](https://github.com/taichi-dev/taichi). 
- developed by Shi-Yihao Zhejiang Universiy(In Progress) 

## Examples

## Features
### Discrete Element Method 
  - only spherical particles supported)

  - Search Algorithm
    1. Sorted based
    2. Multilevel linked cell

### Material Point Method 
  - Constitutive Model
    1. Linear elastic
    2. Von-Mises
    3. Tresca
    4. Mohr-Coumlob
    5. Drucker-Parger
    6. Elas-plastic
    7. Modified Cam Clay
    8. Hardening Soil
    9. Nowtonian Fluid
    10. Non-Newtonian Fluid

  - Integration Scheme
    1. USF
    2. USL
    3. MUSL
    4. GIMPM
    5. MLS-MPM

  - Stability features
    1. B-bar Method
    2. F-bar Method

### DEMPM coupling
  - Coupling Scheme:
    1. Penlaty Method
    2. Incremental potential contact

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
2. Set up environment variables
```
sudo gedit ~/.bashrc
```
3. Add the installation path to bashrc file:
```
export tiDEMPM=/user_path/tiDEMPM
```
4. Run the test
```
python test.py
```

## Acknowledgememt
Implementation is largely inspired by [ComFluSoM](https://github.com/peizhang-cn/ComFluSoM).
