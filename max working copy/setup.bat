:: Create conda environment from yaml file
conda env create --name ci --file environment.yml

echo "Done!"
echo "To activate your virtual environment, use command 'conda activate ci'."
echo "To exit your virtual enviroment, use command 'conda deactivate'."
echo "To run the interface use the command 'python mediapipe-holistic-detection-current-version.py"