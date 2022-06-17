--- How to customize the interactive gestures and code ---

@Jordan, is there a particular colab for 4? Do you think we need to be more explicit? Maybe high-level is enough.

1. Create a training set in TeachableMachine and export the image folders for each class with labels
2. Zip the folders together and upload them to the Pose Classification CoLab Notebook by MediaPipe
3. In the Notebook, generate and export the “poses_csvs_out.csv” file containing the filtered image data with normalized landmark data
4. Load labeled landmark data and pose class labels into a processing CoLab Notebook: shuffle and split data into training and test sets, perform exploratory data analysis (PCA) on pose classes, and train logistic regression classifier. 
5. Export classifier to a .pkl file to be loaded into Python code base
