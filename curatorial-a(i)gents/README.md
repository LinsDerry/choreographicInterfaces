### Curatorial A(i)gents Overview

Curatorial A(i)gents was an exhibition curated by metaLAB (at) Harvard. Half of the screen-based projects relied on the following interactions for which the CI accomplishes through gesture recognition: zoom in/out, scroll up/down, advance right/left, select, span, switch hands, and refresh (see image below). Instructions for how to setup and run this project follow in the next section, as well as more info. for how to adapt the code for your own project.

![Interactive Gestures](/curatorial-a(i)gents/gestures.jpeg)

#### This repository contains code specific to the development of the choreographic interface for metaLAB (at) Harvard's Curatorial A(i)gents exhibition at the Harvard Art Museums' Lighbox Gallery (Spring 2022). 

#### For the exhibition, we refactored Version 2.0 (see code-archive repo), making use of a new choreographic interface module catered to the CI pose sets and interaction needs of the Curatorial A(i)gents projects.

### Setup

The key elements of the CI refactor include the following files and elements:
* CI module .py file designed specifically for the CI requirements
* pose embedding module .py file
* CI main .py file
* logistic regresion .pkl file
* sonification .py file designed for the integration of sound to interactions

1) Open terminal
2) Clone curatorial-a(i)gents repo
3) Check that conda is installed using **conda --version**
4) Create conda environment from .yml file using **conda env create -f CIENV.yml**
5) cd to and run CIMain.py using **python3 CIMain.py**

Authors: Jordan Kruguer, Lins Derry, Maximilian Mueller
* Choreographic Interface Dev Lead - Jordan Kruguer
* Choreographic Poses Dev Lead - Lins Derry
* Sonification Dev Lead - Maximilian Mueller
