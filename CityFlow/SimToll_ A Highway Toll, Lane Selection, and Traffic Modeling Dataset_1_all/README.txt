*** SimToll: A Highway Toll, Lane Selection, and Traffic Modeling Dataset ***
Authors: A. Al-Mousa, R. Alqudah, A. Faza, A
Princess Sumaya University for Technology
Corresponding author: A. Al-Mousa
Contact Information:
A.AlMousa@psut.edu.jo
Princess Sumaya University for Technology- Computer Engineering Department
Amman
Jordan


***General Introduction***
This dataset contains traffic information of a toll highway; the highway consists 
of a toll lane, a carpool lane, and three regular lanes. The traffic information 
includes the number of vehicles on each lane type and the average speed on each 
lane, at intervals of 6 minutes.  The dataset also provides information about the 
individual drivers/vehicles on the highway, like their departure and arrival times
and the lane used. 

The dataset contains a total of 90 different scenarios that cover varying the driver 
population size, the toll price, and the overall percentage of vehicles that are 
eligible to use the carpool lane. 

It is being made public to act as supplementary data for publications and in order
for other researchers to use this data in their own work.

The data in this data set was generated using SUMO traffic simulator.   

***Test equipment***
To test the usage of the dataset, machine learning-based models were built to predict 
whether a driver would arrive late or not at his/her final destination based on his/her
lane choice and the current road conditions. 

***Description of the data in this data set***
The data included in this data set has been organized per driver population size (15K, 
20K and 25K). Three files are available for each population size, Vehicle  info, Traffic
info and Fuzzy info.

Vehicle info file: contains information about each vehicle in the simulation.
Traffic info file: presents information about the traffic on the highway.
Fuzzy info file: contains the information used to determine the lane that the vehicle will
choose to use. 

