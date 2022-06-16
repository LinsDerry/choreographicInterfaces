README for version 1.0

Important things for running the code:

## Windows ##
Assumes you have Python 3 installed.
Assumes that your Python 3 executable is invoked with "python". If that is not the case, you will need to edit the batch script and replace every instance of the "python" command with "python3".
Run the "windows_install.bat" batch script. Don't run as administrator.
Can take ~30 minutes or more depending upon your system and internet connection.


## MacOS ## 
Assumes you have Python 3 installed. (python 3.7.x so that it is compatible with tensorflow)
Navigate to the folder of the repository in your terminal (i.e. wherever you unzip these files)
You need to set up a virtenv for this to work. To do so...
- You will need to make the script executable by running the following command: "sudo chmod +x ./installation_scripts/mac_install.sh"
- Run the shell script with the command: "./installation_scripts/mac_install.sh".
- This installation script also installs the Homebrew package manager.
- Can take ~30 minutes or more depending upon your system and internet connection (shouldn't really take this long)
Now that you have the TMENV setup... to run the code use the following commands:
cd ~/wherever you unzip the files     	# change directory to cloned repo
source TMenv/bin/activate  		# activate venv for Mac/Linux OR
python3 tm_obj_det.py			# executes script, press ctrl+c to quit
deactivate				# to exit the virtual environment
