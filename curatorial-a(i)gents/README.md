## Curatorial A(i)gents Overview

Curatorial A(i)gents was an exhibition curated by metaLAB (at) Harvard. Half of the screen-based projects relied on the following interactions for which the CI accomplishes through gesture recognition: zoom in/out, scroll up/down, advance right/left, select, span, switch hands, and refresh (see image below). Instructions for how to setup and run this project follow in the next section, as well as more info. for how to adapt the code for your own project.

![Interactive Gestures](/curatorial-a(i)gents/gestures.jpeg)

This repository contains code specific to the development of the choreographic interface for metaLAB (at) Harvard's Curatorial A(i)gents exhibition at the Harvard Art Museums' Lighbox Gallery (Spring 2022). 

For the exhibition, we refactored Version 2.0 (see code-archive repo), making use of a new choreographic interface module catered to the CI pose sets and interaction needs of the Curatorial A(i)gents projects.

## Important CI Elements for the Curatorial A(i)gents Project

CIController.py		
CIModule.py		
gestures.jpeg		
sonification-textures
CIENV.yml		README.md		pose-set-pkl-files	sonification.py
CIMain.py		__pycache__		poseEmbedding.py

* ```CIModule.py```: module defining core elements of the CI that can instantiated in project files
* ```poseEmbedding.py```: module defining the pose embedder for processing body landmark input
* ```CIMain.py```: main file determining actions and interactions with CI tools to create a final interface
* ```pose-set-pkl-files```: regression models used for classifying pose input
* ```sonification.py```: sound interaction integrations with CI

## Setup instructions 
1) Open preferred terminal and cd to desired directory
3) Clone curatorial-a(i)gents repo using ```git clone https://github.com/LinsDerry/choreographicInterfaces.git```
4) Check that conda is installed on your machine using ```conda --version```. If installed make sure conda is up to date using ```conda update -n base conda``` If not follow the [install instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#) for your OS. 
6) Create a conda environment from the .yml file using ```conda env create -f ./curatorial-a\(i\)gents/CIENV.yml ```
7) Activate the conda environment using ```TODO```
8) To run the Curatorial A(i)gents version of the CI ```cd``` to and run CIMain.py using ```python CIMain.py```

## Authors:  
* Jordan Kruguer - Choreographic Interface Dev Lead
* Lins Derry - Choreographic Poses Dev Lead
* Maximilian Mueller - Sonification Dev Lead
