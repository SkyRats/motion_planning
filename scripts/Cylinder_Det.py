import numpy as np

res = 60   # Resolução do lidar
r = 25     #[cm] - Raio dos cilindros (Especificado no edital)
d_min = 10 #[cm] - Distancia minima permitida pro Drone chegar perto de algo

sensor = np.array([10,12,10,10,10,10,10,10,10,10,10,10,30,30,34,37,32,36,38,30,30,30,30,30,30,0,0,12,14,17,40,45,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,10,10,10,10,10,10, 10, 11, 13, 11]) # Dado do lidar com tamanho "res"

# Não poderíamos usar definições de variável direto em vez de funções? 

def prop_max ():
    return (1/np.pi)*np.arccos((r+d_min)/(np.sqrt(2*r**2+2*r*d_min+d_min**2)))

def varMax (d_min):
    return np.sqrt(2*r**2+2*r*d_min+d_min**2) - d_min

def varMax_teorico ():
    return np.sqrt(2)*r - r

prop_max = prop_max()
varMax = varMax_teorico()

def detectar ():

    objetos = []    # A lista de objetos que vão ser devolvidas
    dists = sensor[sensor>0]    # Lista ignorando os zeros
    indices = [ i for i in range(res) if sensor[i] != 0 ] # Correspondencia entre indices do array sem zeros e do scan do lidar
    temp = []   # Lista temporaria pra guardar os objetos que vão ser inseridos em "objetos"
    obj_indices = []    # Lista dos indices da lista "dists" que representam os cilindros
    indices_temp = []   # Armazena os indices que vao ser colocados em "obj_indices"

    obj = 0
    temp.append( dists[0] )
    indices_temp.append( indices[0] )
    i = 1   # Comeca de 1 porque analisa os elementos i-1 e i no loop
    # Passa por todos os elementos nao nulos do scan
    while i < len(dists):

        # Vê se o próximo elemento da lista tem uma variação "permitida" de acordo com a fórmula
        if (abs(dists[i] - dists[i-1]) <= varMax): 
            temp.append(dists[i])

        # Se o proximo elemento não for mais um objeto de acordo com varMax, o codigo analisa o tamanho do objeto e faz a proporção pra ver se pode ser um cilindro
        # Nesse caso, seria chão, então reinicializa as listas temporarias
        elif ((len(temp)/res) > prop_max):
            temp = []   
            indices_temp = []
            temp.append(dists[i])
            indices_temp.append( indices[i] )   # indices[i] correponde ao indice "real" do item i da lista não nula

        else:   # Se nenhum dos de cima, então é um cilindro. Ai o codigo bota nas listas "objetos" e "obj_indices" e reinicializa os temporarios
            objetos.append(temp)
            indices_temp.append( indices[i-1] )
            obj_indices.append( indices_temp )

            temp = []
            indices_temp = []
            temp.append(dists[i])
            indices_temp.append( indices[i] )

        i += 1

    # Confere se existe um elemento que passa pela "quebra" do lidar
    # Se existe um elemento nao nulo no final do scan e se o primeiro elemento nao nulo tem indice 0
    if sensor[-1] != 0 and indices[0] == 0:
        
        # Roda pela lista de objetos nao nulos de novo, ate que a distancia entre eles indique que o objeto acabou
        while abs(dists[i% len(dists) ] - dists[ (i-1) % len(dists) ]) <= varMax:
            temp.append(dists[i% len(dists) ])
            i += 1
        if len(temp)/res < prop_max:
            objetos.append(temp)
            indices_temp.append( indices[(i-1) % len(dists)] )
            obj_indices.append(indices_temp)

        """
        Se o objeto na fronteira for valido, entao esse objeto foi quebrado em 2, sendo capturado pela primeira passagem,
           entao o objeto 0 deve ser apagado
        Se o objeto nao for valido, ou seja, se o objeto for grande demais, ele foi quebrado em 2 partes, sendo uma delas detectada pela primeira passagem,
            entao o objeto 0 deve ser apagado
        Assim, so precisamos conferir se o objeto e tao grande que ele nem foi detectado na primeira passagem 
        """

        if  obj_indices[0][0] < indices[(i-1) % len(dists)] < obj_indices[0][1]:
            del objetos[0]
            del obj_indices[0]

    # Se chega so final da lista com um objeto valido armazenado, salva ele na lista
    elif len(temp) > 0 and len(temp)/res < prop_max:
        objetos.append(temp)
        indices_temp.append( indices[i-1] )
        obj_indices.append(indices_temp)
                

    return objetos, obj_indices


objetos, indices = detectar()

print("As distancias dos objetos detectados são: {}".format(objetos))
print("Os intervalos dos indices dos objetos detectados são: {}".format(indices))
print(prop_max*res)
