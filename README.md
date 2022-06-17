# Choreographic Interfaces

The choreographic interfaces repo contains code for installing and running diferrent versions and applications (related projects and installations) of the choreographic interface (CI) on Mac or Windows OS (currently configured for Mac). 

**Repo Contents**
 - **code-archive** - the code archive contains past iterations and prototypes of CI.
 - **ci-utils** - ci utils contains utility code, specifically several google colabs used to generate pose classes and classifiers that are necessary as inputs to the current working version of CI.
 - **curatorial-a(i)gents** - installation scripts and CI code for the Curatorial A(i)gents project, an exhibition curated by metaLAB (at) Harvard.
 - **data-sensorium** - installation scripts and CI code for Data Sensorium, a performance piece developed for the Harvard Art Labs (@Lins maybe add a better description here)


The data-sensorium repo contains code for a subsequent metaLAB (at) Harvard project that utilizes the CI. It is a work-in-progress.

The development-archive repo contains code for previous experiments that used TeachableMachine image model, OpenCV hand tracking, and MediaPipe holistic model to various degrees. The curatorial-a(i)gents repo contains the "best of" these experiments. 

The utility repo includes CoLabs used for customizing the curatorial-a(i)gents project.  

--- How to run the curatorial-a(i)gents project on your computer ---

@Jordan add step-by-step instructions

--- How to customize the interactive gestures and code ---

@Jordan, is there a particular colab for 4? Do you think we need to be more explicit? Maybe high-level is enough.

1. Create a training set in TeachableMachine and export the image folders for each class with labels
2. Zip the folders together and upload them to the Pose Classification CoLab Notebook by MediaPipe
3. In the Notebook, generate and export the “poses_csvs_out.csv” file containing the filtered image data with normalized landmark data
4. Load labeled landmark data and pose class labels into a processing CoLab Notebook: shuffle and split data into training and test sets, perform exploratory data analysis (PCA) on pose classes, and train logistic regression classifier. 
5. Export classifier to a .pkl file to be loaded into Python code base

Note the CoLabs used can be found in the utility rep


## Authors 
 - [Lins Derry](https://github.com/LinsDerry) - Theory, Choreography, CI Code 
 - [Jordan Kruguer](https://github.com/jlkruguer) - CI Code
 - [Maximilian Mueller] - Sonification Code


## TODO:
1. Add more language to readmes very step by step.
2. licensing (Lins - use MIT license)
3. make metalab owner (Lins - transfer ownership)
4. utility code folder for google colab
