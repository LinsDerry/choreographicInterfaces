### Curatorial A(i)gents Overview

Curatorial A(i)gents was an exhibition curated by metaLAB (at) Harvard. Half of the screen-based projects relied on the following interactions for which the CI accomplishes through gesture recognition: zoom in/out, scroll up/down, advance right/left, select, span, switch hands, and refresh (see image below). Instructions for how to setup and run this project follow in the next section, as well as more info. for how to adapt the code for your own project.

![Interactive Gestures](/curatorial-a(i)gents/gestures.jpeg)

#### This repository contains code specific to the development of the choreographic interface for metaLAB (at) Harvard's Curatorial A(i)gents exhibition at the Harvard Art Museums' Lighbox Gallery (Spring 2022). 

#### For the exhibition, we refactored Version 2.0 (see code-archive repo), making use of a new choreographic interface module catered to the CI pose sets and interaction needs of the Curatorial A(i)gents projects.

### Important CI Elements for the Curatorial A(i)gents Project

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
1) Open terminal
2) Clone curatorial-a(i)gents repo
3) Check that conda is installed using **conda --version**
4) Create conda environment from .yml file using **conda env create -f CIENV.yml**
5) cd to and run CIMain.py using **python3 CIMain.py**

Authors: Jordan Kruguer, Lins Derry, Maximilian Mueller
* Choreographic Interface Dev Lead - Jordan Kruguer
* Choreographic Poses Dev Lead - Lins Derry
* Sonification Dev Lead - Maximilian Mueller
