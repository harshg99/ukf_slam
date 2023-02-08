# Unscented Kalman Filter and SLAM with particle filter(master brach has the code)


## Unscented Kalman Filter

### Description

The Unscented Kalman Filter (UKF) is a state estimation algorithm that can be used to estimate the state of a system from noisy measurements. It is an extension of the Kalman filter algorithm that allows for the tracking of nonlinear systems.
In this project, the UKF will be used to estimate the orientation of a robot in three dimensions using data from an IMU, which consists of gyroscopes and accelerometers. The IMU data will be used to update the state estimate of the robot's orientation, while the motion-capture data from the Vicon system will be used for calibration and tuning of the filter.
To implement the UKF, a state vector that represents the orientation of the robot in three dimensions, a process model that describes how the state evolves over time and a measurement model that descirbes the relationship of the IMU data wrt to the state fo the robot is defined


### Results

1. Pitch
![image](https://user-images.githubusercontent.com/28558013/209148358-e6ad91e8-ccde-4440-9515-fb6fd30fb1a0.png)

2. Roll
![image](https://user-images.githubusercontent.com/28558013/209149401-8f0c6707-7b79-4864-9d62-d97b331863e8.png)

3. Yaw
![image](https://user-images.githubusercontent.com/28558013/209149445-582c9802-90ed-49fa-ba81-a725d6bbbd1c.png)


4. Velocity
![image](https://user-images.githubusercontent.com/28558013/209149494-52a48fa9-6689-4e06-be6d-12e1d25cf785.png)


## SLAM

##

An implementation for SLAM using a particle filter. 

### Results


![image](https://user-images.githubusercontent.com/28558013/209156806-9752b013-16de-4040-ae21-e8deee2ec4e7.png)

![image](https://user-images.githubusercontent.com/28558013/209158624-60a8a475-4e9e-4c95-871a-5da36a751460.png)

![image](https://user-images.githubusercontent.com/28558013/209159587-be3b3cfc-03fd-442f-8549-be3050491616.png)

![image](https://user-images.githubusercontent.com/28558013/209160795-f0109244-2a73-45a2-8566-4e1b097b3779.png)






