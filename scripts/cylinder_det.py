import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import LaserScan

from collections import namedtuple
Cylinder = namedtuple('Cylinder', 'cx cy r')

laser_data = LaserScan()

CYLINDER_RADIUS = 0.25 #[m] - Raio dos cilindros (Especificado no edital)
MINIMUM_DISTANCE = 0.15 #[m] - Distancia minima permitida pro Drone chegar perto de algo
DISTANCE_VARIATION_THRESH = 0.1
RADIUS_VARIATION_THRESH = 0.06

def laser_callback(data):
    global laser_data
    laser_data = data

def clear_cylinder(dist, index):
    temp = []
    indices_temp = []
    temp.append(dist)
    indices_temp.append(index)
    return temp, indices_temp

def get_cylinder_candidates(sensor):
    # Parametros de deteccao
    prop_max = (1/np.pi) * np.arccos((CYLINDER_RADIUS + MINIMUM_DISTANCE) / (np.sqrt(2*CYLINDER_RADIUS**2 + 2*CYLINDER_RADIUS*MINIMUM_DISTANCE + MINIMUM_DISTANCE**2)))
    res_min = 3 
    var_max = np.sqrt(2*CYLINDER_RADIUS**2 + 2*CYLINDER_RADIUS*MINIMUM_DISTANCE + MINIMUM_DISTANCE**2) - MINIMUM_DISTANCE

    res = len(sensor)
    sensor = np.where(sensor == np.Inf, 0, sensor)

    dists = sensor[sensor > 0]    # Lista ignorando os zeros
    indices = [ i for i in range(res) if sensor[i] > 0 ] # Correspondencia entre indices do array sem zeros e do scan do lidar

    objetos = []    # A lista de objetos que v o ser devolvidas
    obj_indices = []    # Lista dos indices da lista "dists" que representam os cilindros
    temp = []   # Lista temporaria pra guardar os objetos que v o ser inseridos em "objetos"
    indices_temp = []   # Armazena os indices que vao ser colocados em "obj_indices"

    if len(dists) > 0:

        temp.append( dists[0] )
        indices_temp.append( indices[0] )
        i = 1   # Comeca de 1 porque analisa os elementos i-1 e i no loop
        # Passa por todos os elementos nao nulos do scan
        while i < len(dists):

            # V  se o pr ximo elemento da lista tem uma variacao "permitida" de acordo com a f rmula
            if abs(dists[i] - dists[i-1]) <= var_max and abs(indices[i] - indices[i-1]) == 1:
                temp.append( dists[i] )
                indices_temp.append( indices[i] )

            # Se o proximo elemento nao for mais um objeto de acordo com var_max, o codigo analisa o tamanho do objeto e faz a propor o pra ver se pode ser um cilindro
            # Nesse caso, seria chao, entao reinicializa as listas temporarias
            else:

                if res_min < len(temp) < prop_max*res:   # Se nenhum dos de cima, entao   um cilindro. Ai o codigo bota nas listas "objetos" e "obj_indices" e reinicializa os temporarios
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
                indices_temp.append(indices[previous_i])

                i += 1
                previous_i = i % len(dists)
                current_i = (i-1) % len(dists)

            if (res_min < len(temp) < prop_max*res or np.average(temp) < 0.7):
                objetos.append(temp)
                obj_indices.append( indices_temp )

            """
            Se o objeto na fronteira for valido, entao esse objeto foi quebrado em 2, sendo capturado pela primeira passagem,
            entao o objeto 0 deve ser apagado
            Se o objeto nao for valido, ou seja, se o objeto for grande demais, ele foi quebrado em 2 partes, sendo uma delas detectada pela primeira passagem,
                entao o objeto 0 deve ser apagado
            Assim, so precisamos conferir se o objeto e tao grande que ele nem foi detectado na primeira passagem
            """

            if len(obj_indices[0]) > 1 or obj_indices[0][0] < indices[current_i] < obj_indices[0][1]:
                del objetos[0]
                del obj_indices[0]

        # Se chega so final da lista com um objeto valido armazenado, salva ele na lista
        elif res_min < len(temp) < prop_max*res:
            obj_indices.append(indices_temp)
            objetos.append(temp)
    
    return objetos, obj_indices

def get_cylinders_xy_and_radius(obj, obj_indices):
    cylinders = []
    for i in range(len(obj)):
        cx, cy, r = find_circle_properties(obj[i], obj_indices[i])
        if radii_are_similar(CYLINDER_RADIUS, r):
            cylinders.append( Cylinder(cx, cy, r) )
    return cylinders

def find_circle_properties(obj, obj_indices):
    cx_average = cy_average = r_average = 0

    if len(obj) > 3:
        thetas = indices_to_angles(obj_indices)

        final_index = len(obj)-1 
        x1, y1 = polar_to_cartesian(obj[0], thetas[0])
        x2, y2 = polar_to_cartesian(obj[final_index], thetas[final_index])

        average_index = 1
        centre = len(obj)//2
        distance_from_centre = 0
        i = centre
        is_first_pass = True
        while 0 < i < final_index:
            if i < centre:
                i = centre + distance_from_centre
                distance_from_centre += 1
            else:
                i = centre - distance_from_centre
            
            x3, y3 = polar_to_cartesian(obj[i], thetas[i])
            cx, cy, r = find_circle_xy_and_radius(x1, y1, x2, y2, x3, y3)

            if is_first_pass:
                cx_average = cx
                cy_average = cy
                r_average = r
                is_first_pass = False
                distance_from_centre += 1
                continue

            if (centres_are_close(cx, cy, cx_average, cy_average)
                and radii_are_similar(r, r_average)):
                cx_average = (cx_average*average_index + cx) / (average_index+1)
                cy_average = (cy_average*average_index + cy) / (average_index+1)
                r_average = (r_average*average_index + r) / (average_index+1)
                average_index += 1

    return cx_average, cy_average, r_average

def find_circle_xy_and_radius(x1, y1, x2, y2, x3, y3):
    # Source
    # https://www.xarg.org/2018/02/create-a-circle-out-of-three-points/
    a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2

    b = ( (x1 * x1 + y1 * y1) * (y3 - y2)
        + (x2 * x2 + y2 * y2) * (y1 - y3)
        + (x3 * x3 + y3 * y3) * (y2 - y1) )
 
    c = ( (x1 * x1 + y1 * y1) * (x2 - x3) 
        + (x2 * x2 + y2 * y2) * (x3 - x1) 
        + (x3 * x3 + y3 * y3) * (x1 - x2) )
 
    cx = -b / (2 * a)
    cy = -c / (2 * a)

    r = np.sqrt( (cx-x1)**2 + (cy-y1)**2 )

    return cx, cy, r

def polar_to_cartesian(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def indices_to_angles(obj_indices):
    return np.radians(np.array(obj_indices) - 90)

def centres_are_close(x1, y1, x2, y2):
    return True if np.linalg.norm([x1-x2, y1-y2]) < DISTANCE_VARIATION_THRESH else False

def radii_are_similar(r1, r2):
    return True if abs(r1-r2) < RADIUS_VARIATION_THRESH else False

if __name__ == '__main__':
    rospy.init_node("avoidance")
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan,
        laser_callback, queue_size=1)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        rospy.wait_for_message('/laser/scan', LaserScan)
        cylinders = get_cylinders_xy_and_radius(laser_data)
        print(len(cylinders))
        for cylinder in cylinders:
            print("{0}\t{1}\t\t{2}".format(cylinder.cx, cylinder.cy, cylinder.r))
        print("\n")
        rate.sleep()