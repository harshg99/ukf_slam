import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def average(Y,q):

    #input is Y (7,4) and q(4,)        

    thresh = 0.0001
    f = True
    # print(Y)
    error = np.zeros((7,3)) 
    while(f):
        for i in range(7):

            qj = Y[i,:]
            q_i = Quaternion(qj[0],qj[1:4])
            qi_error = q_i * q.inv()
            #normalize
            # print(qi_error)
            qi_error.normalize()
            
            e_error = qi_error.axis_angle() 
            if np.linalg.norm(e_error) == 0: 
                error[i,:] = np.zeros(3)
            else:
                error[i,:] = (-np.pi + np.mod(np.linalg.norm(e_error) + np.pi, 2 * np.pi)) / np.linalg.norm(e_error) * e_error
        meane = np.mean(error, axis=0)
        # print(np.linalg.norm(meane))

        q_fin = Quaternion()
        q_fin.from_axis_angle(meane)
        q_fin = q_fin * q
        q_fin.normalize()
        q = q_fin

        if np.linalg.norm(meane) < thresh:
            f = False
        # error = np.zeros((7,3))

    # print(q)
    return q,error




def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    # imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    
    # accel = imu['vals'][0:3,:]
    # gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    imu_times = np.array(imu['ts']).T

    # your code goes here
    # np.set_printoptions(threshold=np.inf)

    #accel readings are integers 
    #this is because that there is an ADC that happens
    # accel_roll_x = -accel[0] #imu Ax and Ay are fliiped
    # accel_pitch_y = -accel[1]
    # accel_yaw_z = accel[2]
    # # print(accel_roll_x)
    # accel = np.array([accel_roll_x,accel_pitch_y,accel_yaw_z])

    # gyro_roll_x = gyro[1] 
    # gyro_pitch_y = gyro[2]
    # gyro_yaw_z = gyro[0]

    # gyro = np.array([gyro_roll_x,gyro_pitch_y,gyro_yaw_z])

    # # plt.plot(np.arange(0,accel.shape[1]),gyro_roll_x)
    # # plt.show()
    # plt.plot(accel[0,:])
    # plt.show()    

    # #conversion to metric units 

    # #find out the sens and bias values according to the given data
    # sens_acc = 33.86
    # bias_acc = [511,501,503]

    # sens_gyro = 193.55 #mv/rad/s
    # bias_gyro = [371.5,377,369.5]


    # gyro = gyro.astype(np.float)
    # #all constant operations so can be undertaken by 2d matrix


    # for i in range(3):
    #     for j in range(accel.shape[1]):
    #         accel[i,j] = (float(accel[i,j] - bias_acc[i])*3300)/(1023*sens_acc) #mV/g
    #         # gyro[i,j] = ((gyro[i,j] - bias_gyro[i])*3300*180)/1023*np.pi*sens_gyro #mv/radians/sec
    #         scale = 3300/(1023*sens_gyro)
    #         # print(scale)
    #         diff = float(gyro[i,j] - bias_gyro[i])
    #         # print(diff*scale)
    #         gyro[i,j] = diff*scale #mv/radians/sec
    #         # print(gyro[i,j])

    # plt.plot(accel[0,:])
    # plt.show()


    accel_org = (imu['vals'][0:3,:]).astype('float64')
    gyro_org = (imu['vals'][3:6,:]).astype('float64')
    gyro_org[[0,1,2],:]= gyro_org[[1,2,0], :] # reordered to give x, y, z

    T = np.shape(imu['ts'])[1]

    # your code goes here

    bias_accel = np.array([[511, 501, 503]]) # x, y, z order
    bias_gyro = np.array([[371.5, 377, 369.5]]) # x, y, z order

    alpha_accel = 33.86 #mV/(m/s^2)
    alpha_gyro = 193.55 # mv/(rad/s)

    accel = (accel_org[:,:] - bias_accel.T) * 3300 / (1023*alpha_accel)
    accel[:2,:] *= -1 
    gyro = (gyro_org[:,:] - bias_gyro.T) * 3300 / (1023*alpha_gyro)

    #Comppiling scaled imu values
    imu_new = np.concatenate((accel.T,gyro.T),axis = 1)

    # plt.plot(gyro[0,:])
    # plt.show()

    #----------------
    #calibration code ends here comment before suubmit

    #UKF
    #so we have to do some initializations 
    P = np.identity(3)#(6,6)
    Q = np.identity(3)
    R = np.identity(3)
    roll = np.identity(6)
    #our state space is going to be 7x1 i.e q0,q1,q2,q3,w1,w2,w3
    q = np.array([1,0,0,0])
    w = np.array([0,0,0]) #1x3
    q_pred = q.reshape(-1,1)
    #unit quat
    # print(gyro.shape)
    q = Quaternion(q[0],q[1:])

    for i in range(imu_new.shape[0]): #timesteps
        #get values at each time step
        acc = imu_new[i,:3] 
        gyro = imu_new[i,3:]
        # print(gyro, "gyro")

        #so we have Pk-1, let's get the matrix S
        S = np.linalg.cholesky(P+Q)

        #for calculating the sigma points we need to get the addends
        #check shape
        W = np.concatenate((S*np.sqrt(P.shape[0]),S*-np.sqrt(P.shape[0])),axis = 1) #3x6
        # print(acc)
        # break

        print(W)
        break
        X = np.zeros((6,4)) #sigma mat of all points

        for k in range(2*P.shape[0]): 
            #get the quatermion from addends
            qw_quat = Quaternion()
            qw_quat.from_axis_angle(W[:,k]) #4x1 making quaternion

            
            #getting Xi
            q_mult =  q * qw_quat #1x4
            X[k,0] = q_mult.scalar()
            X[k,1:4] = q_mult.vec()
            # print(X)
            
        X_new = np.zeros(X.shape[1])
        X_new[0] = q.scalar()
        X_new[1:4] = q.vec()

        X_new = X_new.reshape(-1,1)
        # print(X_new.shape)

        X = np.concatenate((X_new.T,X),axis= 0) #7x4 adding mean point
        # print(X.shape)


        #-----------------------------
        #next section



        #need to get timestamps from data
        if i == imu_times.shape[0]-1:
            dt = imu_times[-1]- imu_times[-2]
        else:
            dt = imu_times[i+1]-imu_times[i]



        #process model 
        Y = np.zeros((7,4)) #mat of all points including mean

        q_del = Quaternion()
        q_del.from_axis_angle(gyro*dt) #making quat
        
        for k in range(X.shape[0]): #runs 7 times number of points

            q_Y = Quaternion(X[k,0], X[k,1:4]) 

            Y[k,0] = (q_Y * q_del).scalar()
            Y[k,1:4] = (q_Y * q_del).vec() 


        #now we have obtained Y points
        #new average of objs
        #we get the mean and the error
        



        #-------------------------------------



        xk_bar, residual = average(Y,q)



        #get new covariance 
        #check this
        Pk_bar = np.zeros((3,3))

        for k in range(7):
            Pk_bar += np.outer(residual[k,:],residual[k,:])
        Pk_bar/=7

        #measurement
        g = Quaternion(0,[0,0,1])
        Z = np.zeros((7,3))
        for k in range(Z.shape[0]): #13
            q_z = Quaternion(Y[k,0],Y[k,1:4])

            #rotation from body to world
            #get accelarations and ignore the scaalar value
            
            Z[k,:3] = (q_z.inv() * (g  * q_z)).vec()

        #now mean and error
        Zk_bar = np.mean(Z,axis = 0)
        # print(Zk_bar.shape)
        Zk_bar /= np.linalg.norm(Zk_bar)

        #cov calc
        Pzz = np.zeros((3,3))
        Pxz = np.zeros((3,3))
        Z_diff = Z - Zk_bar #this is the error 
        for k in range(7):
            Pzz += np.outer(Z_diff[k,:],Z_diff[k,:])
            Pxz += np.outer(residual[k,:],Z_diff[k,:])

        Pzz /=2*Pzz.shape[0]+1
        Pxz /=2*Pzz.shape[0]+1

        #incorporating innovations
        # print(acc)
        acc = acc/(np.linalg.norm(acc))
        # Zk_bar = Zk_bar[:3]
        vk = acc- Zk_bar[:3]

        Pvv = Pzz+ R
        # print(Pvv)
        # break
        # print(residual)
        # break
        #computing Kalman gain
        K = np.dot(Pxz,np.linalg.inv(Pvv))
        #updating
        # print(K)
        # break

        q_g = Quaternion()
        q_g.from_axis_angle(K.dot(vk))
        q_new = q_g * xk_bar
        P_new = Pk_bar - K.dot(Pvv).dot(K.T)
        P = P_new
        q = q_new
        # print(q)

     
        new_q = np.array([q.scalar(),q.vec()[0],q.vec()[1],q.vec()[2]])
        q_pred  = np.concatenate((q_pred,new_q.reshape(-1,1)),axis = 1)
        # print(q_pred.shape)


    roll = np.zeros(q_pred.shape[1])
    pitch = np.zeros(q_pred.shape[1])
    yaw = np.zeros(q_pred.shape[1])
    # roll, pitch, yaw are numpy arrays of length T
    #filling valus
    for i in range(q_pred.shape[1]):
        q_done_i = q_pred[:,i]
        q_done = Quaternion(q_done_i[0],q_done_i[1:4])
        roll[i], pitch[i], yaw[i] = q_done.euler_angles()
    return roll,pitch,yaw


if __name__ == "__main__":
    vicon = io.loadmat('vicon/viconRot1'+'.mat')
    obj = Quaternion()

    roll = np.zeros(vicon['rots'].shape[2])
    pitch = np.zeros(vicon['rots'].shape[2])
    yaw = np.zeros(vicon['rots'].shape[2])

    print(vicon['rots'].shape)
    for i in range(vicon['rots'].shape[2]):
        qr = Quaternion()
        # print(vicon['rots'][:,:,i])
        qr.from_rotm(vicon['rots'][:,:,i])
        roll[i], pitch[i], yaw[i] = qr.euler_angles()
    # print(roll)

    pose = estimate_rot()

    plt.plot(pose[0], 'r--')
    plt.plot(roll)
    plt.legend(["estimate","original"])
    plt.title("Roll")
    plt.show()
    plt.plot(pose[1],'r--')
    plt.plot(pitch)
    plt.legend(["estimate","original"])
    
    plt.title("Pitch")
    plt.show()
    plt.plot(pose[2], 'r--')
    plt.plot(yaw)
    plt.legend(["estimate","original"])
    
    plt.title("Yaw")
    plt.show()
    # print("")
    # print("Hi")s