@echo off
echo Installing required packages...
pip install pip virtualenv virtualenvwrapper
echo Creating and activating a fresh virtual environment...
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install wheel
echo Testing virtual environment...
python -m test_environment.py
echo Installing project requirements from requirements.txt...
python -m pip install -r requirements.txt
echo Setup completed successfully!
python -m src\clean\make_dataset.py -i "data\raw" -o "data\processed"
python -m src\analyse\run_analysis.py -i "data\processed" -o "models"
echo Cleaning up...
deactivate
echo Done!