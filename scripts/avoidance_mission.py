import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from MAV import MAV

laser_data = LaserScan()
MAX_VEL = 2
MASK_VELOCITY = 0b0000011111000111

LINEARITY_THRESHOLD = 0.3

# Cylinder detection parameters
r = 0.25     #[m] - Raio dos cilindros (Especificado no edital)
d_min = 0.4  #[m] - Distancia minima permitida pro Drone chegar perto de algo 

def laser_callback(data):
    global laser_data
    laser_data = data
 
def clear_cylinder(dist, index):
    temp = []   
    indices_temp = []
    temp.append(dist)
    indices_temp.append(index)
    return temp, indices_temp

def detect_cylinders():
    global d_min
    global r

    # Parametros de deteccao
    prop_max = (1/np.pi)*np.arccos((r+d_min)/(np.sqrt(2*r**2+2*r*d_min+d_min**2)))
    var_max = np.sqrt(2*r**2+2*r*d_min+d_min**2) - d_min
    
    objetos = []    # A lista de objetos que v o ser devolvidas
    sensor = np.array(laser_data.ranges)
    res = len(sensor)
    sensor = np.where(sensor == np.Inf, 0, sensor)
    
    dists = sensor[sensor > 0]    # Lista ignorando os zeros
    indices = [ i for i in range(res) if sensor[i] > 0 ] # Correspondencia entre indices do array sem zeros e do scan do lidar
        
    temp = []   # Lista temporaria pra guardar os objetos que v o ser inseridos em "objetos"
    obj_indices = []    # Lista dos indices da lista "dists" que representam os cilindros
    indices_temp = []   # Armazena os indices que vao ser colocados em "obj_indices"

    if len(dists) > 0:

        temp.append( dists[0] )
        indices_temp.append( indices[0] )
        i = 1   # Comeca de 1 porque analisa os elementos i-1 e i no loop
        # Passa por todos os elementos nao nulos do scan
        while i < len(dists):

            # V  se o pr ximo elemento da lista tem uma variacao "permitida" de acordo com a f rmula
            if abs(dists[i] - dists[i-1]) <= var_max and abs(indices[i] - indices[i-1]) == 1:
                temp.append(dists[i])

            # Se o proximo elemento nao for mais um objeto de acordo com var_max, o codigo analisa o tamanho do objeto e faz a propor o pra ver se pode ser um cilindro
            # Nesse caso, seria chao, entao reinicializa as listas temporarias
            else:
                
                if 0 < len(temp) < prop_max*res:   # Se nenhum dos de cima, entao   um cilindro. Ai o codigo bota nas listas "objetos" e "obj_indices" e reinicializa os temporarios
                    indices_temp.append( indices[i-1] )
                    
                    objetos.append(temp)
                    obj_indices.append( indices_temp )

                (temp, indices_temp) = clear_cylinder(dists[i], indices[i])
            
            i += 1

        # Confere se existe um elemento que passa pela "quebra" do lidar
        # Se existe um elemento nao nulo no final do scan e se o primeiro elemento nao nulo tem indice 0
        if sensor[-1] != 0 and indices[0] == 0:
            
            # Roda pela lista de objetos nao nulos de novo, ate que a distancia entre eles indique que o objeto acabou
            previous_i = i % len(dists)
            current_i = (i-1) % len(dists)

            while abs(dists[previous_i ] - dists[ current_i ]) <= var_max and i < 2*len(dists) and abs(indices[previous_i] - indices[current_i]) == 1:
                temp.append(dists[previous_i ])
                
                i += 1
                previous_i = i % len(dists)
                current_i = (i-1) % len(dists)
            
            if 0 < len(temp) < prop_max*res:
                indices_temp.append( indices[current_i] )
                
                objetos.append(temp)
                obj_indices.append( indices_temp )

            """
            Se o objeto na fronteira for valido, entao esse objeto foi quebrado em 2, sendo capturado pela primeira passagem,
            entao o objeto 0 deve ser apagado
            Se o objeto nao for valido, ou seja, se o objeto for grande demais, ele foi quebrado em 2 partes, sendo uma delas detectada pela primeira passagem,
                entao o objeto 0 deve ser apagado
            Assim, so precisamos conferir se o objeto e tao grande que ele nem foi detectado na primeira passagem 
            """

            if obj_indices and obj_indices[0][0] < indices[current_i] < obj_indices[0][1]:
                del objetos[0]
                del obj_indices[0]

        # Se chega so final da lista com um objeto valido armazenado, salva ele na lista
        elif 0 < len(temp) < prop_max*res:
                        
            indices_temp.append( indices[i-1] )
            
            obj_indices.append(indices_temp)
            objetos.append(temp)
                    
    return objetos, obj_indices

def saturate(vector):
    norm = np.linalg.norm(vector)
    if norm > MAX_VEL:
        vector = (vector/norm)*MAX_VEL
    return vector

def normal_dist(mean, variance, x):
    return (1/((variance*2*np.pi)**0.5))*np.exp((x-mean)/variance**0.5)

def run():
    rospy.init_node("avoidance")
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan, laser_callback, queue_size=1)
    mav = MAV("1")
    goal = np.array([0, 0])
    initial_height = 1.5
    mav.takeoff(initial_height)

    a=0.004
    b=0.9
    c=0.01
    d=-0.5

    Kr = -4 # repulsive
    Ka = 0.5 # attractive
    Kz = 0.5 # height proportional control
    Ky = -0.5 # yaw proportional control
    mean = 0
    variance = 1.2
    d = (mav.drone_pose.pose.position.x - goal[0])**2 + (mav.drone_pose.pose.position.y - goal[1])**2
    d = np.sqrt(d)
    
        
    while not rospy.is_shutdown() and d > 0.3:
        d = (mav.drone_pose.pose.position.x - goal[0])**2 + (mav.drone_pose.pose.position.y - goal[1])**2
        d = np.sqrt(d)
        
        euler_orientation = euler_from_quaternion(ros_numpy.numpify(mav.drone_pose.pose.orientation))
        ########################theta_goal global###################################
        deltaY = goal[1] - mav.drone_pose.pose.position.y
        deltaX = goal[0] - mav.drone_pose.pose.position.x
        if deltaY > 0 and deltaX >= 0:
            if deltaX == 0:
                theta_goal = 1.57079632679
            else:
                theta_goal = np.arctan(deltaY/deltaX)
        if deltaY >= 0 and deltaX < 0:
            if deltaY == 0:
                theta_goal = 3.14
            else:
                theta_goal = np.arctan(abs(deltaX/deltaY)) + 1.57079632679 #90
        if deltaY < 0 and deltaX <= 0:
            if deltaX == 0:
                theta_goal = -1.57079632679
            else:
                theta_goal = -1*np.arctan(abs(deltaX/deltaY)) - 1.57079632679 #180
        if deltaY <= 0 and deltaX > 0:
            if deltaY == 0:
                theta_goal = 0
            else:
                theta_goal = -1*np.arctan(abs(deltaY/deltaX))
        ##################################################################################
        
        Ft = np.array([0.0, 0.0])
        Fg = np.array([Ka*d*np.cos(theta_goal),
                                    Ka*d*np.sin(theta_goal)])            

        objects, objects_indices = detect_cylinders()

        for i in range(len(objects)):
            middle = (objects_indices[i][0] + objects_indices[i][-1])/2 + 1
            theta = (middle*laser_data.angle_increment) % 2*np.pi
            laser_range = laser_data.ranges[ int(middle) ]  
            
            Fi = Kr * ((a/((laser_range**b)*c)) + d*(laser_range-1.5) - 0.2)
            Fix = -Fi*np.cos(theta + mav.drone_pose.pose.orientation.z)
            Fiy = -Fi*np.sin(theta + mav.drone_pose.pose.orientation.z)
            Ft += np.array([Fix, Fiy])

        Fg = saturate(Fg)
        F = Ft + Fg
        #rospy.loginfo("Attraction = {}; Repulsion = {}".format(Fg, Ft))
        mav.set_position_target(type_mask=MASK_VELOCITY,
                                x_velocity=F[0],
                                y_velocity=F[1],
                                z_velocity=Kz*(initial_height - mav.drone_pose.pose.position.z),
                                yaw_rate= 3)
                                
    mav.land()

run()