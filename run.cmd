@echo off
echo >>> Installing required packages...
pip install pip virtualenv virtualenvwrapper
echo >>> Creating and activating a fresh virtual environment...
python -m venv venv
echo set "OUTDATED_IGNORE=1" >> venv\Scripts\activate.bat
call venv\Scripts\activate.bat
python -m pip install wheel
echo >>> Testing virtual environment...
python test_environment.py
echo >>> Installing project requirements from requirements.txt...
python -m pip install -r requirements.txt
echo >>> Setup completed successfully!
python src\clean\make_dataset.py -i "data\raw" -o "data\processed" -r "references"
python src\analyse\run_analysis.py -i "data\processed" -o "models"
python src\visualise\run_visualisations.py -i "models" -o "reports" -r "references"
echo >>> Build successful! Cleaning up...
deactivate