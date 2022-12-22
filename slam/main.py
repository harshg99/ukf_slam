
import click, tqdm, random

from slam import *
import pdb
def run_dynamics_step(src_dir, log_dir, idx, split, t0=0, draw_fig=False):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx, split)

    # trajectory using odometry (xy and yaw) in the lidar data
    d = slam.lidar
    xyth = []
    for p in d:
        xyth.append([p['xyth'][0], p['xyth'][1],p['xyth'][2]])
    xyth = np.array(xyth)

    plt.figure(1); plt.clf();
    plt.title('Trajectory using onboard odometry')
    plt.plot(xyth[:,0], xyth[:,1])
    logging.info('> Saving odometry plot in '+os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'odometry_%s_%02d.png'%(split, idx)))

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(slam.lidar[0]['xyth'])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)   # maintains all particles across all time steps
    plt.figure(2); plt.clf();
    ax = plt.subplot(111)
    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))

        if draw_fig:
            ax.clear()
            ax.plot(slam.p[0], slam.p[0], '*r')
            plt.title('Particles %03d'%t)
            plt.draw()
            plt.pause(0.01)

    plt.plot(ps[0], ps[1], '*c')
    plt.title('Trajectory using PF')
    logging.info('> Saving plot in '+os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))
    plt.savefig(os.path.join(log_dir, 'dynamics_only_%s_%02d.png'%(split, idx)))

def run_observation_step(src_dir, log_dir, idx, split, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step 
    t0 = 0
    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]
    logging.debug('> Initializing 1 particle at: {}'.format(xyth))
    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx, split):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=100 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.05, Q=np.diag([2e-4,2e-4,1e-4]))
    slam.read_data(src_dir, idx, split)
    T = len(slam.lidar)

    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan. First find the time t0 around which we have both LiDAR
    # data and joint data
    #### TODO: XXXXXXXXXXX
    t0 = 0
    t_slam =slam.lidar[t0]['t']
    t_joint = slam.joint['t'][0]
    while t_slam<t_joint:
        t0+=1
        t_slam =slam.lidar[t0]['t']

    xyth = slam.lidar[t0]['xyth']
    xyth[2] = slam.lidar[t0]['rpy'][2]


    # initialize the occupancy grid using one particle and calling the observation_step
    # function
    #### TODO: XXXXXXXXXXX
    slam = slam_t(resolution=0.05)
    slam.read_data(src_dir, idx, split)

    slam.init_particles(n=1,p=xyth.reshape((3,1)),w=np.array([1]))
    slam.observation_step(t=t0)

    # reinitialise with 100 particles and obtain map using data

    #pdb.set_trace()
    n = 100
    x = np.random.uniform(low=slam.map.xmin/1000,high=slam.map.xmax/1000,size=(n,))
    y = np.random.uniform(low=slam.map.ymin/1000,high=slam.map.ymax/1000,size=(n,))
    yaw = np.random.uniform(low=-0.01,high=0.01,size=(n,))
    p = np.array([x.tolist(),y.tolist(),yaw.tolist()])

    poses = np.zeros(shape=(3,1))
    w = np.ones((n,))/n
    slam.init_particles(n, p, w)
    plt.figure()
    slam.Q = 2e-4*np.eye(3)
    slam.Q[2,2] = 1e-4
    print(T)
    print(t0)

    xyth = []
    for p in slam.lidar:
        xyth.append([p['xyth'][0], p['xyth'][1]])
    xyth = np.array(xyth)
    xyth_actual = slam.map.grid_cell_from_xy(xyth[:,0],xyth[:,1])
    #pdb.set_trace()
    for j in tqdm.tqdm(range(t0,T)):
        slam.dynamics_step(t=j)
        slam.observation_step(t=j)
        max_w = np.argmax(slam.w)
        poses = np.hstack((poses,slam.p[:,[max_w]]))

        slam.resample_particles()
        xyth_pslam = slam.map.grid_cell_from_xy(poses[0,:],poses[1,:])
        #print(j)
        # if(j%5000==0):
        #     map_final = np.zeros((slam.map.cells.shape[0],slam.map.cells.shape[1],3))+1.0
        #     map_final[slam.map.cells==1]= np.array([0,0,0])
        #     map_final[slam.map.num_obs_per_cell==0]= np.array([128,128,128])/255

        #     for k in range(xyth_actual[0].shape[0]):
        #         map_final[xyth_actual[0][k],xyth_actual[1][k],:] = np.array([1.0,0,0])
        #     for k in range(xyth_pslam[0].shape[0]):
        #         map_final[xyth_pslam[0][k],xyth_pslam[1][k],:] = np.array([0,0,1.0])
        #     plt.imshow(map_final,cmap='brg')
        #     plt.savefig(os.path.join(log_dir, 'map_%s_%02d.png'%(split, idx)))
        #     plt.show()

        # plt.imshow(slam.map.cells,cmap='binary')
        # plt.show()
        print("Postion {}".format(slam.p[:,max_w]))
        print("Actual {}".format(slam.lidar[j]['xyth']))
    print("Done")




    # slam, save data to be plotted later
    #### TODO: XXXXXXXXXXX
   

    map_final = np.zeros((slam.map.cells.shape[0],slam.map.cells.shape[1],3))+1.0
    map_final[slam.map.cells==1]= np.array([0,0,0])
    map_final[slam.map.num_obs_per_cell==0]= np.array([128,128,128])/255

    for k in range(xyth_actual[0].shape[0]):
        map_final[xyth_actual[0][k],xyth_actual[1][k],:] = np.array([1.0,0,0])
    for k in range(xyth_pslam[0].shape[0]):
        map_final[xyth_pslam[0][k],xyth_pslam[1][k],:] = np.array([0,0,1.0])
    plt.imshow(map_final,cmap='brg')
    plt.savefig(os.path.join(log_dir, 'map_%s_%02d.png'%(split, idx)))
    plt.show()



@click.command()
@click.option('--src_dir', default='./', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='0', help='dataset number', type=int)
@click.option('--split', default='train', help='train/test split', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
def main(src_dir, log_dir, idx, split, mode):
    # Run python main.py --help to see how to provide command line arguments

    if not mode in ['slam', 'dynamics', 'observation']:
        raise ValueError('Unknown argument --mode %s'%mode)
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)

    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_di4r, idx, split)
        sys.exit(0)
    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx, split)
        sys.exit(0)
    else:
        p = run_slam(src_dir, log_dir, idx, split)
        return p

if __name__=='__main__':
    main()
