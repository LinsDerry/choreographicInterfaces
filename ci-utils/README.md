## How to customize the interactive gestures and code 

The following steps detail how to create a new pose set from scratch that can be used as input to the choregraphc interface.

1. Create a training set in [*TeachableMachine*](https://teachablemachine.withgoogle.com/train/pose) and export the image folders for each class with labels
2. Zip the folders together and upload them to the [*Pose Classification CoLab Notebook*](https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR) by MediaPipe. Be sure to follow the naming and folder structure explained in the "Upload image samples" section. 
4. In the Notebook, generate and export the **fitness_poses_csvs_out.csv** file containing the filtered image data with normalized landmark data. It is not necessary to proceed to "Step 2: Classification", as we will use a different method for classifiying in the next step.
5. Load labeled landmark data and pose class labels into **classifier-generator.ipynb** (in ci-utils folder). This notebook will shuffle and split data into training and test sets, perform exploratory data analysis (PCA) on pose classes, and train a logistic regression classifier to be used by the choreographic interface. The regression model is saved as a .pkl file to be imported by the choreographic interface.
