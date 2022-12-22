import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt
import scipy
import pdb
#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter



def calibrate_acc(data_acc,acc):

    pass

def calibrate_gyro(data_gyro,omega):
    
    pass


def get_sigma_points(mean_quat,mean_vel,var,Q):
    #pdb.set_trace()
    var_sqrt = np.linalg.cholesky(var+Q)
    sigma_points_quat = []
    sigma_points_vel = np.zeros((3,var_sqrt.shape[0]*2+1))
    sigma_points_vel[:,:var_sqrt.shape[0]] = np.expand_dims(mean_vel,axis =1) + np.sqrt(var_sqrt.shape[0])*var_sqrt[3:,:]
    sigma_points_vel[:,var_sqrt.shape[0]:-1] = np.expand_dims(mean_vel,axis=1) - np.sqrt(var_sqrt.shape[0])*var_sqrt[3:,:]
    sigma_points_vel[:,-1] = mean_vel
    qW = Quaternion()
    for j in range(var_sqrt.shape[0]):
        qW.from_axis_angle(np.sqrt(var_sqrt.shape[0])*var_sqrt[0:3,j])
        q_ = mean_quat*qW
        #q_.normalize()
        sigma_points_quat += [q_]
    for j in range(var_sqrt.shape[0]):    
        qW.from_axis_angle(-np.sqrt(var_sqrt.shape[0])*var_sqrt[0:3,j])
        q_ = mean_quat*qW
        #q_.normalize()
        sigma_points_quat += [q_]
    
    sigma_points_quat+=[mean_quat]
    return sigma_points_quat,sigma_points_vel

def propagate(sigma_points_quat,sigma_points_vel,time):
    
    #pdb.set_trace()
    sigma_points_vel_ = sigma_points_vel
    quat_del = Quaternion()
    sigma_points_quat_=[None]*len(sigma_points_quat)
    for j in range(len(sigma_points_quat)):
        quat_del.from_axis_angle(sigma_points_vel[:,j]*time)
        sigma_points_quat_[j] = sigma_points_quat[j]*quat_del
    
    return sigma_points_quat_,sigma_points_vel_

def compute_mean_(sigma_points_quat,sigma_points_vel,mean_quat):
    thresh = 0.0006
    flag = True
    ee = Quaternion()
    #pdb.set_trace()
    error = np.zeros((3,len(sigma_points_quat)))
    while(flag):
        for j in range(len(sigma_points_quat)):
            q = sigma_points_quat[j]*mean_quat.inv()
            q.normalize()
            error_q = np.array([q.axis_angle()])
            if np.linalg.norm(error_q) == 0: # not rotate
                error[:,j] = np.zeros(3)
            else:
                error[:,j] = (-np.pi + np.mod(np.linalg.norm(error_q) + np.pi, 2 * np.pi)) / np.linalg.norm(error_q) * error_q
        
        mean_error = np.mean(error,axis = 1)
        if np.abs(mean_error).sum()<thresh:
            flag = False

        ee.from_axis_angle(mean_error)
        mean_quat = ee*mean_quat
        mean_quat.normalize()
    
    mean_vel = np.mean(sigma_points_vel,axis = 1) 
    err_vel = sigma_points_vel - np.expand_dims(mean_vel,axis=1)
    error_ = np.concatenate([error,err_vel],axis = 0)

    cov = (error_@error_.T)/len(sigma_points_quat)
    return mean_quat,mean_vel,cov,error_

    
def update_meas(mean_q,mean_v,sigma_points_quat,sigma_points_vel,var,accel,gyro,error,R,T):
    g = Quaternion(0,[0,0,1.0])
    Z = np.zeros((var.shape[0],len(sigma_points_quat)))
    #pdb.set_trace()
    for k in range(len(sigma_points_quat)): #13
        q_z = sigma_points_quat[k]
        g_ =  (q_z.inv() * (g  * q_z))
        g_.normalize()
        Z[:3,k] = g_.vec()
        Z[3:,k] = sigma_points_vel[:,k]
    
    Z_mean = np.mean(Z,axis=1)
    
    Z_meas = np.zeros((6,))
    Z_meas[:3] = accel/np.linalg.norm(accel)
    Z_meas[3:] = gyro
    
    P_zz = (Z - np.expand_dims(Z_mean,axis = 1))@(Z.T-np.expand_dims(Z_mean,axis = 0))/len(sigma_points_quat)
    P_xz = error@(Z.T-np.expand_dims(Z_mean,axis = 0))/len(sigma_points_quat)
    P_zz = P_zz+R
    P_zz[0:3,3:] = 0
    P_zz[3:,0:3] = 0
    #P_xz[0:3,3:] = 0
    #P_xz[3:,0:3] = 0

    K = P_xz.dot(np.linalg.inv(P_zz))
    #if T%100 == 0:
    #     print('K')
    #     print(K)
    #     print('V')
    #     print(var)
    #     print('Pzz')
    #     print(P_zz)
    #     print('Pxz')
    #     print(P_xz)
    
    v_k = Z_meas - Z_mean
    K[0:3,3:] = 0
    K[3:,0:3] = 0
    pert = K.dot(v_k)

    eq = Quaternion()
    eq.from_axis_angle(pert[:3])
    mean_q = eq*mean_q
    mean_q.normalize()
    mean_v = mean_v + pert[3:]
    var = var - K.dot(P_zz.dot(K.T))

    return mean_q,mean_v,var        


def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    gyro[[0,1,2],:]= gyro[[1,2,0], :]

    T = np.shape(imu['ts'])[1]

    # your code goes here


    #alpha_acc,bias_acc = calibrate_acc()
    #alpha_gyro,bias_gyro = calibrate_gyro()

    bias_acc = np.expand_dims(np.array([511, 501, 503]),axis = 1)
    alpha_acc = 332.17
    bias_gyro = np.expand_dims(np.array([371.5, 377, 369.5]),axis = 1)
    alpha_gyro = 193.55

    #plt.plot(accel)
    #plt.plot(accel.T)
    #plt.show()
    accel = (accel-bias_acc)*3300/(1023*alpha_acc)
    gyro = (gyro-bias_gyro)*3300/(1023*alpha_gyro)
    accel[:2,:] *= -1

    times = imu['ts'].T
    
    quat = Quaternion()
    vel = np.zeros((3,))
    var = 1.0*np.eye(6)
    Q = 1.0*np.eye(6)
    R = 1.0*np.eye(6)
    rpy = np.zeros((3,T))
    vels = np.zeros((3,T))
    #pdb.set_trace()
    for j in range(T):
        
        if j == times.shape[0]-1:
            dt = times[-1]- times[-2]
        else:
            dt = times[j+1]-times[j]

        sigma_quat,sigma_vel = get_sigma_points(quat,vel,var,Q)
        sigma_quat_prop,sigma_vel_prop = propagate(sigma_quat,sigma_vel,dt)
        quat,vel,var,residual = compute_mean_(sigma_quat_prop,sigma_vel_prop,quat)
        quat,vel,var = update_meas(quat,vel,sigma_quat_prop,sigma_vel_prop,var,accel[:,j],gyro[:,j],residual,R,j)
        rpy[:,j] = quat.euler_angles()
        vels[:,j] = vel
    # roll, pitch, yaw are numpy arrays of length T
    roll = rpy[0,:]
    pitch = rpy[1,:]
    yaw = rpy[2,:]

    #Plotting velocity
    plt.figure()
    plt.title("Estimate vel")
    plt.plot(vels.T, '-')
    #plt.legend()
    # plt.figure()
    plt.plot(gyro.T)
    plt.title("Actual Vel")
    plt.legend(["g_x","g_y","g_z","v_x","v_y","v_z"])
    #plt.show()

    return roll,pitch,yaw
if __name__ == "__main__":
    data_num=3
    r_est,p_est,y_est = estimate_rot(data_num=data_num)
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')

    roll_act = np.zeros(vicon['rots'].shape[2])
    pitch_act = np.zeros(vicon['rots'].shape[2])
    yaw_act = np.zeros(vicon['rots'].shape[2])

    for j in range(vicon['rots'].shape[2]):
        q = Quaternion()
        q.from_rotm(vicon['rots'][:,:,j])
        roll_act[j], pitch_act[j], yaw_act[j] = q.euler_angles()


    plt.figure()
    plt.plot(r_est, '--')
    plt.plot(roll_act)
    plt.title("Roll")
    plt.legend(["est","actual"])

    plt.figure()
    plt.plot(p_est,'--')
    plt.plot(pitch_act)
    plt.title("Pitch")
    plt.legend(["est","actual"])
    
    plt.figure()
    plt.plot(y_est, '--')
    plt.plot(yaw_act)
    plt.title("Yaw")
    plt.legend(["est","actual"])
    plt.show()


