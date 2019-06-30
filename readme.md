# Predicting Loan Grade of Borrowers
This code uses data from the peer to peer lending company Lending Club (formerly Lending Circle). Most work considering this data aims to predict the probability of default, but this work predicts the loan grade assigned to each borrower.

## Requirements
Python 3.x

## Data
The data and data dictionary used can be downloaded here https://data.world/jaypeedevlin/lending-club-loan-data-2007-11

## Usage
Clone the repository to your preferred location
`git clone https://github.com/DataSciencePete/LCGradePrediction.git`
To install the requirements in your own virtual environment using virtualenv
`sudo pip3 install virtualenv`
`virtualenv <name>`
`source activate <path_to_virtualenv>/bin/activate`
To run the models
`python grade_prediction.py <data file> <data dictionary file>`
