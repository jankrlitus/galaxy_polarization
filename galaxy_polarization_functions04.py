from astropy.io import fits # Lectura de ficheros 'fits'
import numpy as np # Funciones matemáticas
import pandas as pd # Proporciona estructuras de datos y herramientas para su análisis
import matplotlib.pyplot as plt # Permite trazar gráficos y figuras en pantalla y guardarlas en un fichero de imagen.
import matplotlib as mtp # Gradaciones de color en gráficos.
import matplotlib.colors as colors # Colores en hexadecimal.
import time # En el desarrollo del programa permite evaluar el tiempo usado en diferentes etapas
from sklearn.cluster import DBSCAN # Búsqueda de 'clusters' 
from scipy.spatial import ConvexHull # Contorno de los clusters hallados
def csv_creator(dataframe:pd.DataFrame,csv_file:str):
    '''
    Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.

    Parametros:
    -----------
    dataframe: DataFrame de 'pandas' con los datos que queremos grabar.
    csv_file: Nombre del fichero csv que se quiere grabar.

    Retorno:
    --------
    No retorna ningun valor. Solo graba en un fichero 'csv' el Dataframe.
    '''
    if csv_file!=None:
        if csv_file[-4:]!=".csv":
            csv_file+=".csv"
        dataframe.to_csv(csv_file,sep=";")

def jpg_creator(image_file:chr=None,dots_per_inch:int=1200):
    '''
    Si 'image_file' contiene un nombre, se graba un fichero imagen '.jpg' con este nombre.

    Parametros:
    -----------
    image_file: Nombre del fichero '.jpg' que se quiere grabar. Si el fichero contiene otra extensión de imagen, se grabará con esta extensión siempre que el módulo 'matplotlib.pyplot.savefig' contemple esta extensión.
    dots_per_inch: Número de 'dpi' de la imagen a guardar. Por defecto se le asignan 1200 dpi.
    
    Retorno:
    --------
    No retorna ningun valor. Solo graba una imagen en el directorio de trabajo.
    '''
    if image_file!=None and image_file!="":
        if image_file[-4]!=".":
            image_file=image_file+".jpg"
        plt.savefig(image_file,bbox_inches='tight',dpi=dots_per_inch)

def deg_to_rad(degrees):
    '''
    Convierte de grados sexagesimales a radianes. El redondeo a 5 decimales lo he realizado para facilitar su lectura en las hojas de cálculo.

    Parametros:
    -----------
    degrees: Grados sexagesimales
    
    Retorno:
    --------
    Grados en radianes redondeados a 5 decimales.
    '''
    return round(degrees*np.pi/180,5)

def weighted_average(values, tolerances):
    '''
    Parametros
    ----------
    values : Array de valores de los que se quiere encontrar la media
    tolerances : Array de tolerancias del array de valores.

    Retorno
    -------
    Devuelve la media ponderada de los valores entregados en función de sus tolerancias.
    Si hay tolerancias inferiores o iguales a 0, se igualan al máximo error encontrado en la zona.
    '''
    values_array=np.array(values) # Convertimos los vectores de Python en arrays de 'numpy'.
    tolerances_array=np.array(tolerances)
    tolerances_array[tolerances_array<=0.0]=max(max(tolerances_array),0.01) # Todas las tolerancias inferiores o iguales a cero se igualan a la máxima tolerancia encontrada (0.01 en caso de que sea 0).
    inv_tol = np.reciprocal(tolerances_array)
    med_pond = np.ma.average(values_array, weights=inv_tol) # Si el array de tolerancias es correcto, hacemos la media de los valores ponderados por la inversa de sus respectivas tolerancias.
    return med_pond

def outliers_by_sigma(polxzone:list,pol_errxzone:list,angxzone:list,ang_errxzone:list,sigma_limit:float):
    '''
    Parametros
    ----------
    polxzone: Lista de porcentajes de polarización de la zona.
    pol_errxzone: Lista de errores de medida esperados en los porcentajes de polarización de la zona.
    angxzone: Lista de angulos de polarización de la zona.
    ang_errxzone: Lista de errores de medida esperados en los angulos de polarización de la zona.
    sigma_limit: Número de sigmas a partir del cual queremos despreciar los puntos.

    Retorno
    -------
    Las mismas listas de parámetros suministradas a la función pero solo con los puntos dentro del número de sigmas establecido.
    polxzone,pol_errxzone,angxzone,ang_errxzone
    '''
    angle_average=weighted_average(angxzone,ang_errxzone)
    angle_sigma=np.std(angxzone)
    
    if angle_sigma==0: #Si la desviación estandar es cero, o bien hay un solo punto o bien los ángulos son exactamente iguales. Por lo tanto retorno los mismos puntos que han entrado.
        polxzone_acum=polxzone
        angxzone_acum=angxzone
        pol_errxzone_acum=pol_errxzone
        ang_errxzone_acum=ang_errxzone
    else:
        angle_min=angle_average-angle_sigma*sigma_limit
        angle_max=angle_average+angle_sigma*sigma_limit
        #Inicializo los acumuladores para los puntos que entren dentro del límite en número de sigmas.
        polxzone_acum=[]
        angxzone_acum=[]
        pol_errxzone_acum=[]
        ang_errxzone_acum=[]
        for n in range(len(polxzone)):
            if angxzone[n]>=angle_min and angxzone[n]<=angle_max:
                polxzone_acum.append(polxzone[n])
                angxzone_acum.append(angxzone[n])
                pol_errxzone_acum.append(pol_errxzone[n])
                ang_errxzone_acum.append(ang_errxzone[n])
    if polxzone_acum==[]: # En caso de que no haya quedado ningún punto, los devolvemos todos
        polxzone_acum=polxzone
        angxzone_acum=angxzone
        pol_errxzone_acum=pol_errxzone
        ang_errxzone_acum=ang_errxzone
    return polxzone_acum,pol_errxzone_acum,angxzone_acum,ang_errxzone_acum

def order_catalog(polarization_data:pd.DataFrame,column_name_1:int,column_name_2:int=None,csv_file=None):
    '''
    Parametros:
    -----------
    polarization_data: DataFrame de los datos de polarización con las columnas: longitude,latitude,polarization,angle,polarization_error,angle_error.
    column_name_1: Nombre de la columna por la que queremos ordenar el DataFrame 'polarization_data'.
    column_name_2: En caso de que se quiera ordenar por una segunda columna indicar su nombre. Si no se indica se ordenará solo por 'column_name_1'.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'
    
    Retorno:
    --------
    Devuelve el DataFrame ordenado por la columna 'column_name_1' o bien por las columnas 'column_name_1' y 'column_name_2'.
    '''
    if column_name_2==None:
        polarization_data=polarization_data.sort_values(column_name_1) #Ordenamos el DataFrame por la columna 'column_name' 
        polarization_data=polarization_data.reset_index(drop=True) #Rehacemos el índice
    else:
        polarization_data=polarization_data.sort_values(by=[column_name_1,column_name_2],ascending=[True,True]) #Ordenamos el DataFrame por la columna 'column_name' 
        polarization_data=polarization_data.reset_index(drop=True) #Rehacemos el índice

    csv_creator(polarization_data,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return polarization_data

def direction_bar_coordinates(lon:float,lat:float,ang:float,bar_length:float):
    '''
    Calcula las coordenadas de los puntos inicial y final de la barra de dirección, siendo su largo el largo de los lados de la zona.

    Parametros:
    -----------
    lon: Longitud del punto (º)
    lat: Latitud del punto (º)
    ang: Angulo de polarización del punto (º)
    bar_length: Longitud de la linea indicativa de la dirección de polarización

    Retorno:
    --------
    x1: Longitud del inicio de la barra (º)
    y1: Latitud del inicio de la barra (º)
    x2: Longitud del final de la barra (º)
    y2: Latitud del final de la barra (º)
    colors: Color 'viridis' indicativo del ángulo de polarización
    '''
    if ang>180:
        ang-=180 # Cuando se representa la perpendicular a la polarización, pueden aparecer direcciones por encima de 180º por lo que lo retrasamos 180º.
    colors=ang/180 #Irá de 0 a 1
    r=bar_length/2
    ang_rad=np.pi*ang/180
    
    x1=lon-r*np.cos(ang_rad)
    x2=lon+r*np.cos(ang_rad)
    y1=lat-r*np.sin(ang_rad)
    y2=lat+r*np.sin(ang_rad)
    return x1,y1,x2,y2,colors

def direction_bar_coordinates_rad(lon:float,lat:float,ang:float,bar_length:float):
    '''
    Calcula las coordenadas de los puntos inicial y final de la barra de dirección, siendo su largo el largo de los lados de la zona.

    Parametros:
    -----------
    lon: Longitud del punto (rad)
    lat: Latitud del punto (rad)
    ang: Angulo de polarización del punto (rad)
    bar_length: Longitud de la linea indicativa de la dirección de polarización

    Retorno:
    --------
    x1: Longitud del inicio de la barra (rad)
    y1: Latitud del inicio de la barra (rad)
    x2: Longitud del final de la barra (rad)
    y2: Latitud del final de la barra (rad)
    colors: Color 'viridis' indicativo del ángulo de polarización
    '''
    if ang>np.pi:
        ang-=np.pi # Cuando se representa la perpendicular a la polarización, pueden aparecer direcciones por encima de 180º por lo que lo retrasamos dichos 180º.
    colors=ang/np.pi #Irá de 0 a 1
    r=bar_length/2
    
    x1=lon-r*np.cos(ang)
    x2=lon+r*np.cos(ang)
    y1=lat-r*np.sin(ang)
    y2=lat+r*np.sin(ang)
    return x1,y1,x2,y2,colors

def cluster_outline(catalog_with_clusters:pd.DataFrame,perpendicular:bool):
    '''
    Parameters:
    -----------
    catalog_with_clusters: DataFrame conteniendo por lo menos las columnas 'longitude','latitude','angle','cluster'
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización. Sirve para que el color del perímetro sea el mismo que el de la barra de dirección.

    Retorno:
    --------
    El contorno de los clusteres contenidos en 'catalog_with_clusters' con el color asociado a la dirección de polarización

    '''
    if perpendicular:
        added_angle=90
    else:
        added_angle=0
    cmap = plt.get_cmap('viridis')
    clusters_number=max(catalog_with_clusters['cluster'])+1
    for clu in range(clusters_number):
        cluster_angle=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['angle'].median()
        cluster_angle+=added_angle
        if cluster_angle>180:
            cluster_angle-=180
        viridis_color=cmap(cluster_angle/180)
        cluster_longitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['longitude']
        cluster_latitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['latitude']
        cluster_coordinates=np.column_stack((cluster_longitude.values,cluster_latitude.values))
        hull=ConvexHull(cluster_coordinates)
        for simplex in hull.simplices:
            plt.plot(cluster_coordinates[simplex, 0], cluster_coordinates[simplex, 1], color=viridis_color,linewidth=0.3)

def cluster_outline_rad(catalog_with_clusters:pd.DataFrame,perpendicular:bool):
    '''
    Parameters:
    -----------
    catalog_with_clusters: DataFrame conteniendo por lo menos las columnas 'longitude','latitude','angle','cluster'. Los ángulos deben estar en radianes para poder usar la proyección Mollweide.
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización. Sirve para que el color del perímetro sea el mismo que el de la barra de dirección.

    Retorno:
    --------
    El contorno de los clusteres contenidos en 'catalog_with_clusters' con el color asociado a la dirección de polarización
    '''
    if perpendicular:
        added_angle=np.pi/2
    else:
        added_angle=0
    cmap = plt.get_cmap('viridis')
    clusters_number=max(catalog_with_clusters['cluster'])+1
    for clu in range(clusters_number):
        cluster_angle=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['angle'].median()
        cluster_angle+=added_angle
        if cluster_angle>np.pi:
            cluster_angle-=np.pi
        viridis_color=cmap(cluster_angle/np.pi)
        cluster_longitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['longitude']
        cluster_latitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['latitude']
        cluster_coordinates=np.column_stack((cluster_longitude.values,cluster_latitude.values))
        hull=ConvexHull(cluster_coordinates)
        for simplex in hull.simplices:
            plt.plot(cluster_coordinates[simplex, 0], cluster_coordinates[simplex, 1], color=viridis_color,linewidth=0.3)

def color_bar_values(number_of_sectors):
    bar_values=[i/number_of_sectors for i in range(number_of_sectors)]+[1.0]
    if 180%number_of_sectors==0:
        bar_labels=[str(int(i*180))+"º" for i in bar_values]
    else:
        bar_labels=[str(round(i*180,1))+"º" for i in bar_values]
    return bar_values,bar_labels

def read_catalog(url="https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/fits?II/226/catalog.dat.gz",head=1,lon_name="GLON",lat_name="GLAT",pol_name="Pol",ang_name="PA",pol_err_name="e_Pol",ang_err_name="e_PA",csv_file=None):
    '''
    Parametros:
    -----------
    Si no se introduce ningún parámetro, la función toma por defecto los del catálogo de Carl Heiles.

    url: Dirección 'web' en donde se encuentra el catálogo en formato 'fits'.
    head: Cabecera en donde se encuentran los datos.
    lon_name: Nombre de la variable de la longitud galáctica.
    lat_name: Nombre de la variable de la latitud galáctica.
    pol_name: Nombre de la variable del porcentaje de polarización.
    ang_name: Nombre de la variable del ángulo de polarización.
    pol_err_name: Nombre de la variable del error estimado del porcentaje de polarización.
    ang_err_name: Nombre de la variable del error estimado del ángulo de polarización.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'

    Retorno:
    --------
    Devuelve un DataFrame de Pandas con los valores de:
     - longitud 'longitude'. En caso de que la longitud vaya de 0º a 360º la modifica de -180º a +180º para que el núcleo galáctico esté en en centro del mapa.
     - latitud 'latitude'
     - porcentaje de polarización 'polarization'
     - dirección de polarización 'angle'
     - error estimado del porcentaje de polarización 'polarization_error'
     - error estimado de la dirección de polarización 'angle_error'
    '''
    #Todo lo que contiene el 'fits'
    all_catalog_data=fits.open(url)
    #Tabla de todos los datos que contiene la cabecera 'head'
    all_data=pd.DataFrame(all_catalog_data[head].data)
    #Subtabla con solo los campos necesarios para generar el mapa de polarización
    polarization_data=all_data[[lon_name,lat_name,pol_name,ang_name,pol_err_name,ang_err_name]]#seleccionamos solo las columnas que nos interesan
    old_columns_name=list(polarization_data.columns)
    new_columns_name=['longitude','latitude','polarization','angle','polarization_error','angle_error']
    #creamos un diccionario en el que la clave son el nombre actual de las columnas 'old_columns_name' y los valores el nuevo nombre 'new_columns_name'.
    changing_names=dict(zip(old_columns_name,new_columns_name))
    polarization_data=polarization_data.rename(columns=changing_names)

    #Si los valores de longitud van de 0º a 360º paso los superiores a 180º a ángulos negativos de forma que el centro del mapa esté situado en el centro galáctico.
    if polarization_data["longitude"].max()>180:
        polarization_data.loc[polarization_data["longitude"]>180,'longitude']=polarization_data.loc[polarization_data["longitude"]>180,'longitude']-360
    csv_creator(polarization_data,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return polarization_data

def space_cutout(polarization_data:pd.DataFrame,lon_min:float,lon_max:float,lat_min:float,lat_max:float,csv_file=None):
    '''
    Devuelve un mapa con los puntos que se hallen dentro del rectángulo definido por (lon_min,lat_min) y (lon_max,lat_max)

    Parametros
    ----------
    polarization_data: DataFrame de los datos de polarización con las columnas: longitude,latitude,polarization,angle,polarization_error,angle_error.
    lon_min: Longitud mínima del submapa a devolver
    lat_min: Latitud mínima del submapa a devolver
    lon_max: Longitud maxima del submapa a devolver
    lat_max: Latitud maxima del submapa a devolver
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'

    Retorno
    -------
    Submapa del 'DataFrame' delimitado por el rectángulo entre la esquina inferior izquierda (lon_min,lat_min) y la superior derecha (lon_max,lat_max).
    '''
    
    for n in range(len(polarization_data)):
        if polarization_data.loc[n,'longitude']>=lon_min and polarization_data.loc[n,'longitude']<=lon_max and polarization_data.loc[n,'latitude']>=lat_min and polarization_data.loc[n,'latitude']<=lat_max:
            pass
        else:
            polarization_data.drop(n,inplace=True)
    polarization_data.reset_index(drop=True,inplace=True) #Debo eliminar el índice para reenumerar las filas. Si no se hace, el índice continua apuntando a la posición de cada punto en el catálogo.
    csv_creator(polarization_data,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return polarization_data

def values_cutout(polarization_data,pol_min,ang_err_max,csv_file=None):
    '''
    Elimina los puntos que estan fuera de los límites indicados.

    Parametros
    ----------
    polarization_data: DataFrame de los datos de polarización con las columnas: longitude,latitude,polarization,angle,polarization_error,angle_error.
    pol_min: Porcentaje de polarización por debajo del cual se eliminan los puntos.
    ang_err_max: Error de ángulo por encima del cual se eliminan los puntos.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'

    Retorno
    -------
    'DataFrame' de la polarización sin los puntos que estén fuera de los límites especificados.
    '''
    for n in range(len(polarization_data)):
        if polarization_data.loc[n,'polarization']>=pol_min and polarization_data.loc[n,'angle_error']<=ang_err_max and polarization_data.loc[n,'angle_error']>0: #Solo elegimos los que tengan un error de ángulo definido 
            pass
        else:
            polarization_data.drop(n,inplace=True)
    polarization_data.reset_index(drop=True,inplace=True) #Debo eliminar el índice para reenumerar las filas. Si no se hace, el índice continua apuntando a la posición de cada punto en el catálogo.
    csv_creator(polarization_data,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return polarization_data

def add_coords_of_zone_center(polarization_data:pd.DataFrame,zone_size:float,csv_file=None):
    '''
    Parametros
    ----------
    polarization_data: 'DataFrame' de 'pandas' con las columnas de datos de polarización. Debe contener las columnas "longitude", "latitude", "polarization", "angle", "polarization_error" y "angle_error".
    zone_size: Dimensión de las zonas en las que queremos dividir el mapa. Por ejemplo si 'zone_size'=3, se dividirá el mapa en zonas de 3°x3°.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'
    
    Retorno
    -------
    El mismo 'DataFrame' de entrada a la función pero añadiendo las columnas 'zone_lon' y 'zone_lat' conteniendo los datos del centro de la zona a la que pertenece cada punto (fila). Este nuevo DataFrame se devuelve ordenado por zonas.
    
    Ejemplo
    -------
    Queremos saber cual es el centro de la zona de 10°x10° que
    contiene el punto (-71.28°,14.09°) considerando que el origen
    del eje 'x' es -180° y el del eje 'y' es -90°.
    el centro de este punto será el (-75,15)

    Es decir:
     (-80,20)     (-70,20)
            ┌─────┐
    (-75,15)┼──·  │
            └─────┘
     (-80,10)     (-70,10)
    '''
    zone_lon_center=[]
    zone_lat_center=[]
    for n in range(len(polarization_data)):
        '''
        Los cuadrantes los identifico con dos valores, el primero (zone_lon_num) es el número de cuadrantes
        que debo desplazarme en longitud (es decir en horizontal) para llegar al que contiene el punto.
        El segundo (zone_lat_num), el número de cuadrantes que debo desplazarme en latitud (es decir
        en vertical) para llegar al cuadrante que contiene el punto.
        El primer cuadrante siempre es el número cero.
        Si calculamos el resultado de las dos ecuaciones siguientes con nuestro ejemplo nos situamos en el
        cuadrante n°3 si nos desplazamos en longitud (horizontalmente) y el n°1 si nos desplazamos en
        latitud (verticalmente)
        '''
        zone_lon_num = int((polarization_data.loc[n,"longitude"]-(-180)) / zone_size) # '-180' es el valor de longitud mas pequeño posible.
        zone_lat_num = int((polarization_data.loc[n,"latitude"]-(-90)) / zone_size) # '-90' es el valor de latitud mas pequeño posible.
        
        '''
        Ahora tan solo tenemos que calcular cual es el centro de este cuadrante (zone).
        En nuestro caso el centro del cuadrante (3,1) es el (-145,-75)
        '''
        zone_lon_center.append(zone_lon_num * zone_size + (-180) + zone_size / 2) # '-180' es el valor de longitud mas pequeño posible.
        zone_lat_center.append(zone_lat_num * zone_size + (-90) + zone_size / 2) # '-90' es el valor de latitud mas pequeño posible.
        
    polarization_data['zone_lon']=zone_lon_center
    polarization_data['zone_lat']=zone_lat_center
    csv_creator(polarization_data,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return polarization_data.sort_values(['zone_lon','zone_lat'],ascending=[True,True],ignore_index=True)

def statistics_per_zone(polarization_data_with_zones:pd.DataFrame,sigma_limit:float,csv_file=None):
    '''
    Parametros
    ----------
    polarization_data_with_zones: 'DataFrame' de 'pandas'. Debe contener la longitud (zone_lon) y la latitud (zone_lat) del centro de la zona a la que pertenece cada punto. Asimismo las filas deben estar ordenadas primero por 'zone_lon' y segundo por 'zone_lat'.
    sigma_limit: Número de sigmas a partir del cual queremos despreciar los puntos.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'
    
    Retorno
    -------
    'DataFrame' con los centros de cada zona ('zone_lon', 'zone_lat'), su porcentaje y ángulo de polarización ('zone_pol', 'zone_ang') y los puntos por zona antes y después de la seleccion por número de sigmas ('points_before', 'points_after').
    '''
    polarization_dataxzone=pd.DataFrame() # DataFrame vacio
    polxzone=[polarization_data_with_zones.loc[0,'polarization']] # iniciamos los vectores de acumulacion con el primer valor de la primera zona
    pol_errxzone=[polarization_data_with_zones.loc[0,'polarization_error']]
    angxzone=[polarization_data_with_zones.loc[0,'angle']] 
    ang_errxzone=[polarization_data_with_zones.loc[0,'angle_error']]
    current_longitude=polarization_data_with_zones.loc[0,'zone_lon']
    current_latitude=polarization_data_with_zones.loc[0,'zone_lat']
    for n in range(1,len(polarization_data_with_zones)): # Puesto que ya hemos inicializado los vectores de acumulación con el primer valor, empezamos desde el segundo (el 1).
        if polarization_data_with_zones.loc[n,'zone_lon']==current_longitude and polarization_data_with_zones.loc[n,'zone_lat']==current_latitude:
            polxzone.append(polarization_data_with_zones.loc[n,'polarization']) # Añadimos al vector 'polxzone' un nuevo valor de la zona en estudio
            pol_errxzone.append(polarization_data_with_zones.loc[n,'polarization_error']) # Añadimos al vector 'pol_errxzone' un nuevo valor de la zona en estudio
            angxzone.append(polarization_data_with_zones.loc[n,'angle']) # Añadimos al vector 'angxzone' un nuevo valor de la zona en estudio
            ang_errxzone.append(polarization_data_with_zones.loc[n,'angle_error']) # Añadimos al vector 'ang_errxzone' un nuevo valor de la zona en estudio
        else:
            points_before=len(polxzone)
            polxzone,pol_errxzone,angxzone,ang_errxzone=outliers_by_sigma(polxzone,pol_errxzone,angxzone,ang_errxzone,sigma_limit) # Eliminamos los puntos que tengan un angulo fuera del limite de sigmas establecido
            points_after=len(polxzone)
            average_polarizationxzone=weighted_average(polxzone,pol_errxzone) # Hemos encontrado el primer punto de una nueva zona por lo que calculamos las medias de los valores encontrados en la zona anterior
            average_anglexzone=weighted_average(angxzone,ang_errxzone)
            new_zone=pd.DataFrame({'zone_lon':[current_longitude],'zone_lat':[current_latitude],'zone_pol':[average_polarizationxzone],'zone_ang':[average_anglexzone],'points_before':[points_before],'points_after':[points_after]})
            polarization_dataxzone=pd.concat([polarization_dataxzone,new_zone]) # Añadimos los valores encontrados para la zona anterior al DataFrame 'polarization_dataxzone'
            polxzone=[polarization_data_with_zones.loc[n,'polarization']] # iniciamos de nuevo los vectores de acumulacion con el primer valor de esta nueva zona
            pol_errxzone=[polarization_data_with_zones.loc[n,'polarization_error']]
            angxzone=[polarization_data_with_zones.loc[n,'angle']] 
            ang_errxzone=[polarization_data_with_zones.loc[n,'angle_error']]
            current_longitude=polarization_data_with_zones.loc[n,'zone_lon'] # Nueva posición para la nueva zona
            current_latitude=polarization_data_with_zones.loc[n,'zone_lat']
    
    polxzone,pol_errxzone,angxzone,ang_errxzone=outliers_by_sigma(polxzone,pol_errxzone,angxzone,ang_errxzone,sigma_limit) # Eliminamos los puntos que tengan un angulo fuera del limite de sigmas establecido
    average_polarizationxzone=weighted_average(polxzone,pol_errxzone) # Hemos encontrado el primer punto de una nueva zona por lo que calculamos las medias de los valores encontrados en la zona anterior
    average_anglexzone=weighted_average(angxzone,ang_errxzone)
    new_zone=pd.DataFrame({'zone_lon':[current_longitude],'zone_lat':[current_latitude],'zone_pol':[average_polarizationxzone],'zone_ang':[average_anglexzone],'points_before':[points_before],'points_after':[points_after]})
    polarization_dataxzone=pd.concat([polarization_dataxzone,new_zone]) # Añadimos los valores encontrados para la zona anterior al DataFrame 'polarization_dataxzone'
    polarization_dataxzone=polarization_dataxzone.reset_index(drop=True)
    csv_creator(polarization_dataxzone,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.        
    return polarization_dataxzone

def cartesian_plot_by_zones(zone_polarization_data:pd.DataFrame,zone_size:float,perpendicular:bool,image_file:chr=None):
    '''
    Esta función crea una representación de los vectores de polarización por zonas.
    La polarización se representa mediante una línea con el ángulo de la polarización media de cada zona
    y el color (en la escala de colores 'viridis') indica el nivel de polarización en la zona.

    Parametros:
    -----------
    zone_polarization_data: 'DataFrame' de 'pandas' con el porcentaje y ángulo de polarización de cada zona.
    zone_size: tamaño en grados de la zona
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización.
    image_file: Si se introduce un nombre, se graba una imagen con este nombre en el directorio de trabajo.
    
    Retorno:
    --------
    No retorna ninguna variable. 
    Se retorna una imagen en la ventana de representación de Python y, si se ha introducido el nombre de un fichero,
    se graba una imagen con este nombre en el directorio de trabajo.
    '''

    if perpendicular:
        added_angle=90
    else:
        added_angle=0
    
    #Establece una gradacion de colores 'viridis' en formato hexadecimal para poder utilizarlo en matplotlib.
    #viridis = cm.get_cmap('viridis', 256)
    viridis = mtp.colormaps.get_cmap('viridis')
    
    #Para determinar el color para el vector de cada zona, necesitamos que el valor de la polarización en todas las zonas esté entre 0 y 1. Para hacerlo necesitamos tener el mínimo y el máximo de polarización.
    pol_min=min(zone_polarization_data.loc[:,'zone_pol'])
    pol_max=max(zone_polarization_data.loc[:,'zone_pol'])
    
    #Puesto que también puede representarse un submapa, debemos ver cuales son los límites en longitud y latitud del 'DataFrame'
    lon_min=int(min(zone_polarization_data.loc[:,'zone_lon']))-zone_size/2
    lat_min=int(min(zone_polarization_data.loc[:,'zone_lat']))-zone_size/2
    lon_max=int(max(zone_polarization_data.loc[:,'zone_lon']))+zone_size/2
    lat_max=int(max(zone_polarization_data.loc[:,'zone_lat']))+zone_size/2

    #Fijamos el ancho y alto del grafico (en pulgadas)
    if lon_max-lon_min>lat_max-lat_min:
        plt.figure(figsize=(10,10*(lat_max-lat_min)/(lon_max-lon_min)))
    else:
        plt.figure(figsize=(10*(lon_max-lon_min)/(lat_max-lat_min),10))
    
    #Establece el color de fondo de la figura
    plt.gca().set_facecolor('black')

    #Establecemos los límites del mapa
    plt.xlim(lon_min,lon_max)
    plt.ylim(lat_min,lat_max)

    for pos in range(len(zone_polarization_data.loc[:,'zone_lon'])):
        '''
        Leemos los parámetros de la zona n°'pos'
        '''
        lonz=zone_polarization_data.loc[pos,'zone_lon']
        latz=zone_polarization_data.loc[pos,'zone_lat']
        angz=zone_polarization_data.loc[pos,'zone_ang']+added_angle
        
        polz=(zone_polarization_data.loc[pos,'zone_pol']-pol_min)/(pol_max-pol_min) #Para dar un color a la flecha en función de la escala de colores 'viridis' ajustamos el valor de la polarización a un valor entre 0 y 1.
        
        x1,y1,x2,y2,color=direction_bar_coordinates(lonz,latz,angz,zone_size) #Coordenadas de inicio y final de la barra de dirección.
        '''
        Valor del color segun la escala 'viridis' para un valor 'polz' entre 0 y 1.
        '''
        color = viridis(polz)
        hex_color = colors.to_hex(color)
        '''
        Dibujamos la flecha para cada zona
        '''
        plt.plot((x1,x2),(y1,y2),hex_color,linewidth=polz) # Color y grosor de la barra de dirección en función del porcentaje de polarizacion
        
    '''
    salvamos el gráfico en un fichero de imagen y también lo representamos en 
    el 'plot' de Python
    '''
    if image_file!=None and image_file!="":
        if image_file[-4]!=".":
            image_file=image_file+".jpg"
        plt.savefig(image_file,bbox_inches='tight',dpi=1200)
    jpg_creator(image_file)
    plt.show()

def mollweide_plot_by_zones(zone_polarization_data:pd.DataFrame,zone_size:float,perpendicular:bool,image_file:chr=None):
    '''
    Esta función crea una representación de los vectores de polarización por zonas.
    La polarización se representa mediante una línea con el ángulo de la polarización media de cada zona
    y el color (en la escala de colores 'viridis') indica el nivel de polarización en la zona.

    Parametros:
    -----------
    zone_polarization_data: 'DataFrame' de 'pandas' con el porcentaje y ángulo de polarización de cada zona.
    zone_size: tamaño en grados de la zona
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización.
    image_file: Si se introduce un nombre, se graba una imagen con este nombre en el directorio de trabajo.
    
    Retorno:
    --------
    No retorna ninguna variable. 
    Se retorna una imagen en la ventana de representación de Python y, si se ha introducido el nombre de un fichero,
    se graba una imagen con este nombre en el directorio de trabajo.
    '''
    if perpendicular:
        added_angle=90
        chosen_color=(0,0,1,1) #Azul
    else:
        added_angle=0
        chosen_color=(1,0,0,1) #Rojo
    
    # Crear una figura
    plt.figure()

    # Crear un subplot con la proyección de Mollweide
    plt.subplot(111, projection="mollweide")

    # Añadir una cuadrícula
    plt.grid(True,color='black',linewidth=0.1)
    plt.tick_params(axis='x',colors='black',labelsize=2)
    plt.tick_params(axis='y',colors='black',labelsize=2)
    
    #Para determinar el grosor de la barra de dirección, necesitamos que el valor de la polarización en todas las zonas esté entre 0 y 1. Para hacerlo necesitamos tener el mínimo y el máximo de polarización.
    pol_min=(min(zone_polarization_data.loc[:,'zone_pol']))
    pol_max=(max(zone_polarization_data.loc[:,'zone_pol']))
    
    #Establece el color de fondo de la figura
    plt.gca().set_facecolor('white')
    
    for pos in range(len(zone_polarization_data.loc[:,'zone_lon'])):
        '''
        Leemos los parámetros de la zona n°'pos'
        '''
        lonz=zone_polarization_data.loc[pos,'zone_lon']
        latz=zone_polarization_data.loc[pos,'zone_lat']
        angz=(zone_polarization_data.loc[pos,'zone_ang']+added_angle)
        '''
        Para dar un color a la flecha en función de la escala de colores 'viridis'
        ajustamos el valor de la polarización a un valor entre 0 y 1.
        '''
        polz=((zone_polarization_data.loc[pos,'zone_pol'])-pol_min)/(pol_max-pol_min)
        '''
        La longitud de la flecha no puede superar los límites de la zona.
        '''
        x1,y1,x2,y2,color=direction_bar_coordinates(lonz,latz,angz,zone_size) #Coordenadas de inicio y final de la barra de dirección.
        x1_rad=deg_to_rad(x1)
        y1_rad=deg_to_rad(y1)
        x2_rad=deg_to_rad(x2)
        y2_rad=deg_to_rad(y2)

        '''
        Dibujamos la barra de dirección para cada zona
        '''
        if polz>0.2:
            plt.plot((x1_rad,x2_rad),(y1_rad,y2_rad),color=chosen_color,linewidth=polz) # linewidth => Grosor de la barra
    '''
    salvamos el gráfico en un fichero de imagen y también lo representamos en 
    el 'plot' de Python
    '''
    jpg_creator(image_file)
    plt.show()

def cluster_catalog_creator(catalog:pd.DataFrame,max_separation:float,min_num_points:int,csv_file=None):
    '''
    Parametros:
    -----------
    catalog: DataFrame conteniendo 'longitude', 'latitude', 'polarization', 'angle', 'polarization_error', 'angle_error'.
    max_separation: separación máxima entre dos puntos para considerar que están dentro de un mismo clúster. Por ejemplo 3.5º.
    min_num_points: mínimo número de puntos vecinos para considerar que un punto forma parte de un clúster. Por ejemplo 10.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'
        
    Retorno:
    --------
    Devuelve un nuevo DataFrame con solo los puntos que pertenecen a algun cluster de cada sector así como el ángulo medio del cluster al que pertenecen.
    '''
    catalog_3D=catalog.drop(columns=['polarization','polarization_error','angle_error']) # Dejo solo las columnas 'longitude', 'latitude', 'angle'
    clusters=DBSCAN(eps=max_separation,min_samples=min_num_points).fit_predict(catalog_3D) # Busco los clusters a partir de estas tres dimensiones. 'clusters' es una lista de python.
    catalog_with_clusters=pd.DataFrame({'longitude':catalog_3D['longitude'],'latitude':catalog_3D['latitude'],'angle':catalog_3D['angle'],'cluster':clusters[:]}) # Nuevo DataFrame incluyendo el nº de cluster
    catalog_with_clusters=catalog_with_clusters[catalog_with_clusters['cluster']>=0] # Eliminamos los puntos con un valor de cluster igual a -1, es decir, los puntos que no pertenecen a ningun cluster.
    catalog_with_clusters=catalog_with_clusters.sort_values(by=['cluster','longitude','latitude'],ascending=[True,True,True]) #Ordenamos el DataFrame por la columna 'cluster' 
    catalog_with_clusters=catalog_with_clusters.reset_index(drop=True) #Rehacemos el indice
    csv_creator(catalog_with_clusters,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return catalog_with_clusters

def clusters_center(catalog_with_clusters:pd.DataFrame,csv_file=None):
    '''
    Parametros:
    -----------
    catalog_with_clusters: DataFrame conteniendo como mínimo 'longitude', 'latitude', 'angle' y 'cluster'.
    csv_file: Si se ha introducido un nombre, graba una hoja de cálculo con este nombre en formato 'csv'

    Retorno:
    --------
    Devuelve un nuevo DataFrame con solo las coordenadas de cada cluster ('longitude' y 'latitude') calculadas como el centro de gravedad de los puntos del cluster, así como su ángulo medio ('angle').
    '''
    clusters_number=max(catalog_with_clusters['cluster'])+1
    clusters_catalog=pd.DataFrame()
    for clu in range(clusters_number): #Iremos seleccionando las coordenadas y el ángulo medio de cada cluster.
        cluster_angle=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['angle'].median() #Angulo medio de los puntos de cada cluster.
        cluster_longitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['longitude'].mean() #Longitudes de los puntos de cada cluster.
        cluster_latitude=catalog_with_clusters[catalog_with_clusters['cluster']==clu]['latitude'].mean() #Latitudes de los puntos de cada cluster.
        new_cluster=pd.DataFrame({'longitude':[cluster_longitude],'latitude':[cluster_latitude],'angle':[cluster_angle]})
        clusters_catalog=pd.concat([clusters_catalog,new_cluster]) #Vamos añadiendo las coordenadas del centro de gravedad de cada cluster y su ángulo medio..
    clusters_catalog=clusters_catalog.reset_index(drop=True) #Rehacemos el índice.
    csv_creator(clusters_catalog,csv_file) #Si 'csv_file' contiene un nombre, se graba un fichero 'csv' con este nombre.
    return clusters_catalog

def cartesian_plot_3D_clusters(catalog_with_clusters:pd.DataFrame,clusters_catalog:pd.DataFrame,perpendicular:bool,image_file:chr=None):
    '''
    Parametros:
    -----------
    catalog_with_clusters: DataFrame de pandas. Debe contener al menos longitud 'longitude', latitud 'latitude' y angulo de polarización de su sector 'angle' y el número de cluster 'cluster'.
    clusters_catalog:  DataFrame de pandas. Debe contener al menos la longitud 'longitude' y latitud 'latitude' del centro de gravedad de cada cluster así como el ángulo medio del cluster al que pertenece.
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización.
    image_file: Si se introduce un nombre, se graba una imagen con este nombre en el directorio de trabajo.
    
    Retorno:
    --------
    Muestra el mapa del plano galáctico con la dirección de polarización del sector al que pertenece así como el perímetro del cluster.
    '''
    if perpendicular:
        added_angle=90
    else:
        added_angle=0
    plt.figure(figsize=(13,5)) #Relación de aspecto del mapa
    plt.gca().set_facecolor('black') #Color de fondo del mapa
    #establecemos los límites del gráfico
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    # Crear un mapa de colores 'viridis'
    cmap = plt.get_cmap('viridis')
    plt.xlabel("Galactic longitude (°)")
    plt.ylabel("Galactic latitude (°)")
    #Dibujamos la cuadrícula. Si utilizo el 'grid' de matplotlib no acepta ni la relación de aspecto ni el fondo de color
    for y in range(-80,81,10):
        plt.plot((-180,180),(y,y),'white',linewidth=0.1,linestyle=(30, (10, 20))) #Cuadrícula horizontal (lineas de 10 separadas 20)
    
    for x in range(-180,181,10):
        plt.plot((x,x),(-90,90),'white',linewidth=0.1,linestyle=(0, (10, 20))) #Cuadrícula vertical (lineas de 10 separadas 20)
    # Dibujar la dirección de cada cluster
    
    for n in range(len(clusters_catalog)):
        x1,y1,x2,y2,colors=direction_bar_coordinates(clusters_catalog.loc[n,'longitude'],clusters_catalog.loc[n,'latitude'],clusters_catalog.loc[n,'angle']+added_angle,8)
        plt.scatter(clusters_catalog.loc[n,'longitude'], clusters_catalog.loc[n,'latitude'], color=cmap(colors),marker="o",s=0.1)
        plt.plot((x1,x2),(y1,y2),color=cmap(colors),linewidth=0.25)
    
    cluster_outline(catalog_with_clusters,perpendicular)
    plt.title("Polarization direction")
    cbar=plt.colorbar()
    bar_values,bar_labels=color_bar_values(6)
    # Establecer las posiciones de las etiquetas
    cbar.set_ticks(bar_values)
    # Establecer las etiquetas
    cbar.set_ticklabels(bar_labels)
    jpg_creator(image_file)
    plt.show()

def mollweide_plot_3D_clusters(catalog_with_clusters:pd.DataFrame,clusters_catalog:pd.DataFrame,perpendicular:bool,image_file=None):
    '''
    Parametros:
    -----------
    catalog_with_clusters: DataFrame de pandas. Debe contener al menos longitud 'longitude', latitud 'latitude' y angulo de polarización de su sector 'angle' y el número de cluster 'cluster'.
    clusters_catalog:  DataFrame de pandas. Debe contener al menos la longitud 'longitude' y latitud 'latitude' del centro de gravedad de cada cluster así como el ángulo del sector al que pertenece.
    number_of_sectors: Número de sectores utilizados para deteminar los puntos que pertenecen a un cluster. Se usa para poder indicar la toleráncia de ángulo usada para seleccionar los puntos que pertenecen a un cluster.
    perpendicular: Si es 'True' nos representa la perpendicular, es decir la dirección del campo magnético asociado a esta polarización.
    image_file: Si se introduce un nombre, se graba una imagen con este nombre en el directorio de trabajo.
    
    Retorno:
    --------
    Muestra el mapa del plano galáctico con la dirección de polarización del sector al que pertenece así como el perímetro del cluster.
    '''
    '''
    Para trabajar con la proyección Mollweide necesitamos que los ángulos estén en radianes.
    Para ello modificamos tanto la longitud y latitud como el ángulo de polarización.
    '''
    if perpendicular:
        added_angle=np.pi/2
    else:
        added_angle=0
    
    catalog_with_clusters['longitude']=deg_to_rad(catalog_with_clusters['longitude'])
    catalog_with_clusters['latitude']=deg_to_rad(catalog_with_clusters['latitude'])
    catalog_with_clusters['angle']=deg_to_rad(catalog_with_clusters['angle'])

    clusters_catalog['longitude']=deg_to_rad(clusters_catalog['longitude'])
    clusters_catalog['latitude']=deg_to_rad(clusters_catalog['latitude'])
    clusters_catalog['angle']=deg_to_rad(clusters_catalog['angle'])

    # Crear una figura
    plt.figure()

    # Crear un subplot con la proyección de Mollweide
    plt.subplot(111, projection="mollweide")
    
    # Color de fondo
    plt.grid(True,color='black',linewidth=0.1)

    #Color y tamaño de las etiquetas de los ejes de longitud y latitud
    plt.tick_params(axis='x',colors='black',labelsize=4)
    plt.tick_params(axis='y',colors='black',labelsize=4)
    
    # Color de fondo
    plt.gca().set_facecolor('white') #Color de fondo del mapa
    
    #Gradación de colores según 'viridis'
    cmap = plt.get_cmap('viridis')

    # Dibujar la dirección de cada cluster
    for n in range(len(clusters_catalog)):
        x1,y1,x2,y2,colors=direction_bar_coordinates_rad(clusters_catalog.loc[n,'longitude'],clusters_catalog.loc[n,'latitude'],clusters_catalog.loc[n,'angle']+added_angle,0.1)
        plt.scatter(clusters_catalog.loc[n,'longitude'], clusters_catalog.loc[n,'latitude'], color=cmap(colors),marker="o",s=0.1)
        plt.plot((x1,x2),(y1,y2),color=cmap(colors),linewidth=0.25)
    
    cluster_outline_rad(catalog_with_clusters,perpendicular)
    cbar=plt.colorbar()
    bar_values,bar_labels=color_bar_values(6)
    # Establecer las posiciones de las etiquetas
    cbar.set_ticks(bar_values)
    # Establecer las etiquetas
    cbar.set_ticklabels(bar_labels)
    cbar.ax.tick_params(labelsize=6)# Cambiamos el tamaño de las etiquetas de la barra de colores
    if image_file!=None:
        plt.savefig(image_file,bbox_inches='tight',dpi=1200)
    plt.show()

if __name__=="__main__":
    task=input('''
1.- Por zonas y grafico en cartesianas
2.- Por zonas y grafico en proyección Mollweide
3.- Clustering 3D y grafico en cartesianas
4.- Clustering 3D y grafico en proyección Mollweide
? ''')
    polarization_data=read_catalog(csv_file="0_initial_catalog.csv") #Lectura del catálogo
    print('Finalizada la lectura del catálogo')
    task=int(task)
    if task==1 or task==2:
        zone_size=3
        sigma_limit=3
        longitude_min=-180
        longitude_max=180
        latitude_min=-90
        latitude_max=90
        pol_min=0.1
        ang_err_max=45
        perpendicular=False
        print(f'''
zone_size={zone_size}\t\tsigma_limit={sigma_limit}
longitude_min={longitude_min}\tlongitude_max={longitude_max}
latitude_min={latitude_min}\tlatitude_max={latitude_max}
pol_min={pol_min}\t\tang_err_max={ang_err_max}
perpendicular={perpendicular}''')
        polarization_data=space_cutout(polarization_data,longitude_min,longitude_max,latitude_min,latitude_max)
        polarization_data=values_cutout(polarization_data,pol_min,ang_err_max)
        polarization_data=add_coords_of_zone_center(polarization_data,zone_size,"1_catalog_with_zones.csv")
        polarization_data=order_catalog(polarization_data,'zone_lon','zone_lat',"2_catalog_ordered.csv")
        polarizationxzones=statistics_per_zone(polarization_data,sigma_limit,'3_polarizationxzones.csv')
        print('Finalizado el cálculo por zonas')
        if task==1:
            print('Iniciando el trazado por zonas del mapa en cartesianas')
            cartesian_plot_by_zones(polarizationxzones,zone_size,perpendicular,"4_cartesian_plot_by_zones")
        else:
            print('Iniciando el trazado por zonas del mapa en proyección Mollweide')
            mollweide_plot_by_zones(polarizationxzones,zone_size,perpendicular,"4_mollweide_plot_by_zones")
    elif task==3 or task==4:
        print('Iniciando el trazado por clusters (2D) del mapa en cartesianas')
        longitude_min=-180
        longitude_max=180
        latitude_min=-90
        latitude_max=90
        pol_min=0.25
        ang_err_max=30
        maximum_distance_between_neighbors=4
        minimum_number_of_neighbors=20
        perpendicular=True
        print(f'''
longitude_min={longitude_min}\tlongitude_max={longitude_max}
latitude_min={latitude_min}\tlatitude_max={latitude_max}
pol_min={pol_min}\t\tang_err_max={ang_err_max}
maximum_distance_between_neighbors={maximum_distance_between_neighbors}
minimum_number_of_neighbors={minimum_number_of_neighbors}
perpendicular={perpendicular}
''')
        polarization_data=space_cutout(polarization_data,longitude_min,longitude_max,latitude_min,latitude_max)
        polarization_data=values_cutout(polarization_data,pol_min,ang_err_max)
        catalog_with_clusters=cluster_catalog_creator(polarization_data,maximum_distance_between_neighbors,minimum_number_of_neighbors,"1_catalog_with_clusters.csv") #Sustituye el ángulo de cada punto por el ángulo del sector y añade el nº de cluster al que pertenece (los puntos que no pertenecen a ningun cluster son eliminados)
        clusters_catalog=clusters_center(catalog_with_clusters,"2_clusters_catalog.csv") #Crea un nuevo DataFrame conteniendo las coordenadas del centro de gravedad de cada cluster y el ángulo del sector asociado.
        if task==3:
            cartesian_plot_3D_clusters(catalog_with_clusters,clusters_catalog,perpendicular,"3_cartesian_plot_3D_cluster") #Presenta el mapa de polarización conteniendo la dirección por sectores de los puntos y de cada cluster.
        else:
            mollweide_plot_3D_clusters(catalog_with_clusters,clusters_catalog,perpendicular,"3_mollweide_plot_3D_cluster") #Presenta el mapa de polarización conteniendo la dirección por sectores de los puntos y de cada cluster.