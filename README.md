# ti-DEMPM 
A high performance objected-oriented Discrete Element Method- Material Point Method (DEM-MPM) simulator in [Taichi](https://github.com/taichi-dev/taichi). 
- developed by Shi-Yihao Zhejiang Universiy(In Progress) 

## Examples
### Sphere impacts and penetrates into granular material
<p align="center">
  <img src="https://github.com/Yihao-Shi/ti-DEMPM/blob/main/Validation/DEMPM/animation3_1.gif" width="50%" height="50%" />
</p>
<p align="center">
  <img src="https://github.com/Yihao-Shi/ti-DEMPM/blob/main/Validation/DEMPM/result3_1.gif" width="50%" height="50%" />
</p>

### Stone Column using ti-DEMPM
<p align="center">
  <img src="https://github.com/Yihao-Shi/ti-DEMPM/blob/main/demo/result.gif" width="50%" height="50%" />

## Features
### Discrete Element Method 
  - only spherical particles supported)

  - Search Algorithm
    1. Sorted based
    2. Multilevel linked cell

### Material Point Method 
  - Constitutive Model
    1. Linear elastic
    2. Drucker-Parger
    3. Nowtonian Fluid

  - Integration Scheme
    1. USF
    2. USL
    3. MUSL
    4. GIMPM

  - Stability features
    1. B-bar Method
    2. F-bar Method

### DEMPM coupling
  - Coupling Scheme:
    1. Penlaty Method
    2. Barrier Method

## Future Work
  1. Multisphere particles (clump) 
  2. GPU memory allocate
  3. More constitutive models in MPM
  4. Pre-processing (i.e., input obj files and JSON)

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
