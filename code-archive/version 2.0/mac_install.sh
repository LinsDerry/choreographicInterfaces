#!/bin/bash
# Install script for OpenCV & MediaPipe for Mac

echo "This script will update your packages, and install new packages to run OpenCV and MediaPipe."
echo "It can take up to an ... to run depending upon your system."
echo "Upon completion, you may need to reboot your machine before running."
echo "Python 3 should be installed before running script."

conda create -n choreographic-interface
echo "Created conda environment --> choreographic-interface"

conda install opencv
pip3 install mediapipe
 
echo "Done!"
echo "To activate your conda environment, use command 'conda activate choreographic-interface'."
echo "To exit your conda enviroment, use command 'conda deactivate'."