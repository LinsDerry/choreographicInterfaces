## How to customize the interactive gestures and code 

1. Create a training set in TeachableMachine and export the image folders for each class with labels
2. Zip the folders together and upload them to the Pose Classification CoLab Notebook by MediaPipe
4. In the Notebook, generate and export the “poses_csvs_out.csv” file containing the filtered image data with normalized landmark data
5. Load labeled landmark data and pose class labels into classifier-generator.ipynb. This notebook will shuffle and split data into training and test sets, perform exploratory data analysis (PCA) on pose classes, and train a logistic regression classifier to be used by the choreographic interface. The regression model is saved as a .pkl file to be inported by the choreographic interface.


@LINS - please add hyperlinks to items 1 and 2.
