#!/bin/bash

# Check if pip is installed
if ! command -v pip &> /dev/null
then
    echo "pip could not be found. Please install pip first."
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found. Creating file..."
    cat << EOF > requirements.txt
pandas == 2.2.2
requests == 2.31.0
matplotlib == 3.5.3
seaborn == 0.13.2
geopandas == 0.13.2
scikit-learn == 1.1.0
Basemap == 1.4.1
pycountry == 24.6.1
pywaffle == 1.1.1
squarify == 0.4.3
joypy == 0.2.6
calmap == 0.0.11
ipywidgets == 8.0.2
EOF
    echo "requirements.txt created successfully."
fi

# Install packages from requirements.txt
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "Installation complete."