import rospy
import numpy as np
import ros_numpy
from sensor_msgs.msg import LaserScan

laser_data = LaserScan()

CYLINDER_RADIUS = 0.25 #[m] - Raio dos cilindros (Especificado no edital)
MINIMUM_DISTANCE = 0.15 #[m] - Distancia minima permitida pro Drone chegar perto de algo
DISTANCE_THRESH = 0.05
LENGTH_THRESH = 0.05

def laser_callback(data):
    global laser_data
    laser_data = data
    cylinders, indices = detect_cylinders(laser_data)
    print(len(cylinders))
    print()

def clear_cylinder(dist, index):
    temp = []
    indices_temp = []
    temp.append(dist)
    indices_temp.append(index)
    return temp, indices_temp

def detect_cylinders(laser_data):
    global MINIMUM_DISTANCE
    global CYLINDER_RADIUS

    # Parametros de deteccao
    prop_max = (1/np.pi) * np.arccos((CYLINDER_RADIUS + MINIMUM_DISTANCE) / (np.sqrt(2*CYLINDER_RADIUS**2 + 2*CYLINDER_RADIUS*MINIMUM_DISTANCE + MINIMUM_DISTANCE**2)))
    var_max = np.sqrt(2*CYLINDER_RADIUS**2 + 2*CYLINDER_RADIUS*MINIMUM_DISTANCE + MINIMUM_DISTANCE**2) - MINIMUM_DISTANCE

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
                temp.append( dists[i] )
                indices_temp.append( indices[i] )

            # Se o proximo elemento nao for mais um objeto de acordo com var_max, o codigo analisa o tamanho do objeto e faz a propor o pra ver se pode ser um cilindro
            # Nesse caso, seria chao, entao reinicializa as listas temporarias
            else:

                if 0 < len(temp) < prop_max*res and object_is_a_circle(temp, indices_temp):   # Se nenhum dos de cima, entao   um cilindro. Ai o codigo bota nas listas "objetos" e "obj_indices" e reinicializa os temporarios
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

            if (0 < len(temp) < prop_max*res or np.average(temp) < 0.7) and object_is_a_circle(temp, indices_temp):
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
        elif 0 < len(temp) < prop_max*res and object_is_a_circle(temp, indices_temp):
            obj_indices.append(indices_temp)
            objetos.append(temp)

    return objetos, obj_indices

def object_is_a_circle(obj, obj_indices):
    cx_average = cy_average = r_average = 0

    if len(obj) > 3:
        thetas = np.multiply(obj_indices, laser_data.angle_increment)
        - laser_data.angle_min

        for i in range(len(obj)-3):
            x1, y1 = polar_to_cartesian(obj[i], thetas[i])
            x2, y2 = polar_to_cartesian(obj[i+1], thetas[i+1])
            x3, y3 = polar_to_cartesian(obj[i+2], thetas[i+2])

            cx, cy, r = find_circle_xy_and_radius(x1, y1, x2, y2, x3, y3)
            if i == 0:
                cx_average = cx
                cy_average = cy
                r_average = r
                continue

            if (centres_are_close(cx, cy, cx_average, cy_average)
                and radii_are_similar(r, r_average)):
                cx_average = (cx_average*i + cx) / (i+1)
                cy_average = (cy_average*i + cx) / (i+1)
                r_average = (r_average*i + cx) / (i+1)
                pass
            else:
                return False

        print(cx_average, cy_average, r_average)
        return True

    return False

def find_circle_xy_and_radius(x1, y1, x2, y2, x3, y3):
    # From
    # https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    sx13 = pow(x1, 2) - pow(x3, 2)
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
        (x12) + (sx21) * (x13) +
        (sy21) * (x13)) // (2 *
        ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
        (sx21) * (y13) + (sy21) * (y13)) //
        (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
        2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    square_of_r = h * h + k * k - c

    # r is the radius
    r = np.sqrt(square_of_r)

    return h, k, r

def polar_to_cartesian(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def centres_are_close(x1, y1, x2, y2):
    return True if np.linalg.norm([x1-x2, y1-y2]) < DISTANCE_THRESH else False

def radii_are_similar(r1, r2):
    return True if abs(r1-r2) < LENGTH_THRESH else False

if __name__ == '__main__':
    rospy.init_node("avoidance")
    laser_sub = rospy.Subscriber("/laser/scan", LaserScan,
        laser_callback, queue_size=1)
    rospy.spin()