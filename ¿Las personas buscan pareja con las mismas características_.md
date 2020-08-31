# ¿Las personas buscan pareja con las mismas características?

- objetivo: Analizar un conjunto de datos para poder determinar el comportamiento de los matrimonios, cuáles son las características que mayor los identifican y resolver esta pregunta a través de Visualización de Machine Learning.

- Requisitos
-Instalar Anaconda para hacer uso de Jupyter Notebook/O usar colab si esta relacionado.
- Obtener los datos desde la plataforma: http://sistemas.inec.cr/pad5/index.php/catalog/253/data-dictionary/F14?file_name=MATRI2018


# Descargar Anaconda
Ingresamos a la Pagina de [Anaconda](tps://www.anaconda.com/) nos dirigimos a la parte de [Download](https://www.anaconda.com/products/individual) (descargas)

Atención: Elegir la versión de Python 3.8  y seleccionar el instalador Gráfico (Graphical Installer)

### Iniciar y Actualizar Anaconda
Vamos a comprobar que todo se haya instalado correctamente y verificar la version más reciente, iniciamos Anaconda Navigator donde podremos lanzar Jupyter Notebooks! Que en este caso es con el vamos a trabajar. 
Para comprobar la instalación vamos a la terminal o linea de comandos windows.
```sh
1 conda -V
```
y obtenemos la versión
```sh
1 python -V
```
y verificamos la versión de Python de nuestro sistema.
```sh
1 conda update anaconda
```

### Conjunto de datos:
Para este caso se tomaron datos desde el 2014 al 2018, este conjunto de datos tiene un total de 128,243 y 33 atributos.
Según los datos obtenidos de la página del INEC (Instituto Nacional de Estadística y Censos), en el año 2018, se consensuaron un total de 23,603 matrimonios. Dentro de ellos se describieron a su vez 33 variables, de los cuales los más importantes se destacan:
1. La mayoría de los hombres se casa a los 28 años, mientras que, la minoría a los 18. La mayoría de las mujeres se casan a los 27 años de edad, mientras que, la minoría se casa después de los 40 años.
2. El mes que registra más casos de matrimonios es diciembre para ambos sexos y los meses que menos registran es septiembre y octubre para ambos sexos.
3. El estado conyugal que tienen ambos sexos antes de casarse es soltero, en la mayoría de los casos registrados. Por otro lado, el estado viudo son aquellos que tienen menor cantidades de matrimonios a comparación con los demás estados conyugales.
4. La mayor cantidad de matrimonios son en la provincia de San José. mientras que, la minoría de casos fue en la provincia de Guanacaste para ambos sexos en el 2018.
5. Por último, la ocupación de la mayoría de los hombres que se casan es de “Trabajadores de los servicios y vendedores de comercios y mercados”. Para las mujeres es “Administradora de hogar”; Sin embargo, la ocupación con menor tendencia a casarse es “Directores y gerentes” o personas “desempleadas”, esto según la fuente de base de datos del INEC que corresponden al momento del matrimonio.

### Descripción de los datos
- Nombre:Directorio de Matrimonios 2018,Directorio de Matrimonios 2017,Directorio de Matrimonios 2016,Directorio de Matrimonios 2015,Directorio de Matrimonios 2014
- Fuente/Creador: INE Costa Rica
- Proceso de creación:
. Forma de acceso:
- Ingresar: http://sistemas.inec.cr/pad5/index.php/catalog/253
Para poder acceder a los datos, se tuvo que solicitar acces direcctamente, aceptar terminos y condiciones.
- Licencia: Estos datos fueron solicitados del Instituto Nacional de Estadística y Censos,donde se tuvo que llenar una descripción para el uso que se le iba a dar al dataset,  
- Fecha de publicación/Fecha de recuperación: 2018,2017,2016,2015,2014
- Formato de los archivos: .sav 


### Diccionario de Datos
### Variables / Atributos :
| Variables |  |
| ------ | ------ |
| anotrab | Año de trabajo |
| mestrab | Mes de trabajo |
| clasemat  | Clase de matrimonio |
| provocu | Provincia de ocurrencia |
| pcocu  | Cantón de ocurrencia |
|pcdocu   | Distrito de la ocurrencia |
| mesmat  | Mes que se realiza el matrimonio |
|anomat   |Año que se realiza el matrimonio  |
|edadhom  |  Edad del hombre|
| edhomrec  |  Edad del hombre en grupos|
| paishom  | País de origen del hombre |
|nachom   | Nacionalidad del hombre |
| grocuhom  | Ocupación del hombre en grupos |
| provhom  | Provincia de residencia del hombre |
|pchom   | Cantón de residencia del hombre |
| pcdhom  | Distrito de residencia del hombre |
| iuhom  | Índice de urbanidad del hombre |
| escivhom  | Estado conyugal del hombre |
|numathom   | Número de matrimonios del hombre |
| edadmuj | Edad de la mujer |
|edmujrec   |Edad de la mujer en grupos  |
|  paismuj | País de origen de la mujer |
|nacmuj   |Nacionalidad de la mujer  |
|grocumuj   |Ocupación de la mujer en grupos  |
|provmuj   | Provincia de residencia de la mujer |
|iumuj   | Índice de urbanidad de la mujer |
|escivmuj | Estado conyugal de la mujer |
|numatmuj   |  Número de matrimonios de la mujer|
| diadeclara  |Día que se realiza la declaración  |
|mesdeclara  | Mes que se realiza la declaración |
|anodeclara  | Año que se realiza la declaración |



### Exploración de datos:

El primer paso es importar pandas y cargar los dataset, en esta ocasión se observa que, no son csv sino .sav (extensión genérica que se utiliza para guardar los archivos y datos).
```sh
import pandas as pd
```
Leer dataset de matrimonios
```sh
df2018=pd.read_spss("MATRIMONIOS2018.sav")
```
```sh
df2017=pd.read_spss("MATRI2017.sav")
```
```sh
df2017=pd.read_spss("MATRI2016.sav")
```
```sh
df2017=pd.read_spss("MATRI2015.sav")
```
```sh
df2017=pd.read_spss("MATRI2014.sav")
```
```sh
frames = [df2018, df2016, df2015 , df2014, df2017]
df = pd.concat(frames)
df= df.reset_index(drop=True)
df
```

Crear un dataframe para hombres
```sh
df_hombres = df[['anotrab', 'clasemat', 'edadhom', 'edhomrec', 'paishom', 'nachom', 'grocuhom', 'provhom', 'pchom', 'pcdhom', 'iuhom', 'escivhom', 'numathom','índice_mat']]
df_hombres = df_hombres.rename(columns={'anotrab': 'año', 'clasemat': 'clase_mat', 'edadhom': 'edad', 'edhomrec': 'edad_rango', 'paishom': 'pais', 'nachom': 'nacionalidad', 'grocuhom': 'ocupación', 'provhom': 'provincia', 'pchom': 'cantón','pcdhom': 'distrito', 'iuhom': 'índice_urbanidad', 'escivhom': 'estado_civil', 'numathom': '#_mat' })
df_hombres
```
Crear un dataframe para mujeres
Renombrar columnas
Renombrar categorías, por ejemplo soltera ... soltero
```sh
df_mujeres = df[['anotrab', 'clasemat', 'edadmuj','edmujrec', 'paismuj', 'nacmuj', 'grocumuj', 'provmuj', 'pcmuj', 'pcdmuj', 'iumuj', 'escivmuj', 'numatmuj','índice_mat']]
df_mujeres = df_mujeres.rename(columns={'anotrab': 'año', 'clasemat': 'clase_mat', 'edadmuj': 'edad','edmujrec': 'edad_rango', 'paismuj': 'pais', 'nacmuj': 'nacionalidad', 'grocumuj': 'ocupación', 'provmuj': 'provincia', 'pcmuj': 'cantón','pcdmuj': 'distrito', 'iumuj': 'índice_urbanidad', 'escivmuj': 'estado_civil', 'numatmuj': '#_mat' })
df_mujeres = df_mujeres.replace('Soltera', 'Soltero')
df_mujeres = df_mujeres.replace('Casada civil', 'Casado civil')
df_mujeres = df_mujeres.replace('Soltera', 'Soltero')
df_mujeres = df_mujeres.replace('Divorciada', 'Divorciado')
df_mujeres = df_mujeres.replace('Viuda', 'Viudo')
df_mujeres
```
Concatenar datasets de hombres y mujeres
```sh
frames = [df_hombres, df_mujeres]
df_total = pd.concat(frames)
df_total
```
```sh
df_total['ocupación'] = df_total['ocupación'].str.replace("Trabajadoras de los servicios y vendedoras de comercios y mercados","Trabajadores de los servicios y vendedores de comercios y mercados")
df_total['ocupación'] = df_total['ocupación'].str.replace("Pensionada","Pensionado")
df_total['ocupación'] = df_total['ocupación'].str.replace("Operadoras de instalaciones y máquinas y ensambladoras","Operadores de instalaciones y máquinas y ensambladores")
df_total['ocupación'] = df_total['ocupación'].str.replace("Directoras y gerentes","Directores y gerentes")
df_total['ocupación'] = df_total['ocupación'].str.replace("Oficiales, operarias y artesanas de artes mecánicas y de otros oficios","Oficiales, operarios y artesanos de artes mecánicas y de otros oficios")
df_total['ocupación'] = df_total['ocupación'].str.replace("Agricultoras y trabajadoras calificadas agropecuarias, forestales y pesqueras 	","Agricultores y trabajadores calificados agropecuarios, forestales y pesqueros")
df_total['ocupación'] = df_total['ocupación'].str.replace("Desempleada","Desempleado")
```
```sh
df_total.info()
```
```sh
column_names = ['edad_rango', 'pais', 'nacionalidad','ocupación', 'provincia', 'cantón', 'distrito','índice_urbanidad', 'estado_civil', '#_mat']
comparación = pd.DataFrame(columns = column_names)
comparación['edad_rango'] = df_total['edad_rango'].iloc[0:128243] == df_total['edad_rango'].iloc[128243:256486]
comparación['pais'] = df_total['pais'].iloc[0:128243] == df_total['pais'].iloc[128243:256486]
comparación['nacionalidad'] = df_total['nacionalidad'].iloc[0:128243] == df_total['nacionalidad'].iloc[128243:256486]
comparación['ocupación'] = df_total['ocupación'].iloc[0:128243] == df_total['ocupación'].iloc[128243:256486]
comparación['provincia'] = df_total['provincia'].iloc[0:128243] == df_total['provincia'].iloc[128243:256486]
comparación['cantón'] = df_total['cantón'].iloc[0:128243] == df_total['cantón'].iloc[128243:256486]
comparación['distrito'] = df_total['distrito'].iloc[0:128243] == df_total['distrito'].iloc[128243:256486]
comparación['índice_urbanidad'] = df_total['índice_urbanidad'].iloc[0:128243] == df_total['índice_urbanidad'].iloc[128243:256486]
comparación['estado_civil'] = df_total['estado_civil'].iloc[0:128243] == df_total['estado_civil'].iloc[128243:256486]
comparación['#_mat'] = df_total['#_mat'].iloc[0:128243] == df_total['#_mat'].iloc[128243:256486]
comparación
```
```sh
comparación['total'] = 10 - comparación[column_names].sum(axis=1)
comparación
```
```sh
comparación = comparación.replace(True,1)
comparación = comparación.replace(False,0)
comparación
```
Se observa los atributos de los matrimonios, el cual, lo más relevante es la nacionalidad y ocupación. Se denomina que, los atributos con mayor diferencia es de ocupación y nacionalidad. Después se redundan los atributos de 10 a 7, dado que, habían muchos atributos que se asemejan y producían ruido en el dataset.
```sh
comparación[["edad_rango", "pais", "nacionalidad", "ocupación", "provincia", "cantón", "distrito", "índice_urbanidad", "estado_civil", "#_mat"]].describe(
```
Importamos las siguientes librerias
se observa una gráfica comparativa de datos. Se encuentran varias características en una escala de 0 a 10, que representa los matrimonios que comparten intereses en común, donde 0 son los matrimonios que comparten muchos intereses en común y 10 es el caso contrario, matrimonios comparten pocos intereses en común. Según la gráfica se observa que, de 0 a 5,000 matrimonios, comparte el 100% de sus intereses; de 0 a 25,000 comparten un 80%. Casi la mayoría de matrimonios en Costa Rica comparte un 60% de su interés y se denota que, los matrimonios comparten menos interés (20%) y contribuye a pocos matrimonios.
```sh
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
```
```sh
sns.distplot(comparación['total'], bins=11, kde=False, rug=True);
```
En esta parte se busca comparar el rango de edad de hombres versus mujeres, restando la edad del hombre menos la de mujer, luego se indica que, si esa diferencia es mayor o igual a 5, que ponga un “true”, significa que son muy similares; si es mayor de 5 años que pongo un “false”.
```sh
column_names = ['edad_rango', 'nacionalidad','ocupación', 'distrito','índice_urbanidad', 'estado_civil', '#_mat']
comparación = pd.DataFrame(columns = column_names)
comparación['edad_rango'] = df_total['edad_rango'].iloc[0:128243] == df_total['edad_rango'].iloc[128243:256486]
comparación['nacionalidad'] = df_total['nacionalidad'].iloc[0:128243] == df_total['nacionalidad'].iloc[128243:256486]
comparación['ocupación'] = df_total['ocupación'].iloc[0:128243] == df_total['ocupación'].iloc[128243:256486]
comparación['distrito'] = df_total['distrito'].iloc[0:128243] == df_total['distrito'].iloc[128243:256486]
comparación['índice_urbanidad'] = df_total['índice_urbanidad'].iloc[0:128243] == df_total['índice_urbanidad'].iloc[128243:256486]
comparación['estado_civil'] = df_total['estado_civil'].iloc[0:128243] == df_total['estado_civil'].iloc[128243:256486]
comparación['#_mat'] = df_total['#_mat'].iloc[0:128243] == df_total['#_mat'].iloc[128243:256486]
comparación
```
```sh
comparación['total'] = 7 - comparación[column_names].sum(axis=1)
comparación
```
```sh
sns.distplot(comparación['total'], bins=8, kde=False, rug=True);
```
```sh
column_names = ['edad_dif', 'nacionalidad','ocupación', 'distrito','índice_urbanidad', 'estado_civil', '#_mat']
comparación = pd.DataFrame(columns = column_names)
comparación['nacionalidad'] = df_total['nacionalidad'].iloc[0:128243] == df_total['nacionalidad'].iloc[128243:256486]
comparación['ocupación'] = df_total['ocupación'].iloc[0:128243] == df_total['ocupación'].iloc[128243:256486]
comparación['distrito'] = df_total['distrito'].iloc[0:128243] == df_total['distrito'].iloc[128243:256486]
comparación['índice_urbanidad'] = df_total['índice_urbanidad'].iloc[0:128243] == df_total['índice_urbanidad'].iloc[128243:256486]
comparación['estado_civil'] = df_total['estado_civil'].iloc[0:128243] == df_total['estado_civil'].iloc[128243:256486]
comparación['#_mat'] = df_total['#_mat'].iloc[0:128243] == df_total['#_mat'].iloc[128243:256486]


df_total['edad'] = pd.to_numeric(df_total['edad'], errors='coerce')
comparación['edad'] = df_total['edad'].iloc[0:128243] - df_total['edad'].iloc[128243:256486]

comparación.loc[comparación['edad'] <= 5, 'edad_dif'] = True
comparación.loc[comparación['edad'] > 5, 'edad_dif'] = False

comparación = comparación.drop(columns=['edad'])

comparación
```
```sh
comparación['total'] = 7 - comparación[column_names].sum(axis=1)
comparación
```
```sh
comparación = comparación.replace(True,1)
comparación = comparación.replace(False,0)
comparación
```
```sh
comparación[["edad_dif", "nacionalidad", "ocupación", "distrito", "índice_urbanidad", "estado_civil", "#_mat"]].describe()
```
```sh
sns.distplot(comparación['total'], bins=8, kde=False, rug=True);
```
Analizar cada grupo por separado
```sh
df['total']=comparación['total']
```
Puede Observar los grupos de 0 que no hay diferencias: 34% son católicos. El porcentaje de matrimonio civil entre más diferencia tiende aumentar la cantidad de matrimonios civil, mientras que, los católicos es el contrario, entre menos diferencias tienen, mayor es el índice de matrimonio. El porcentaje de católicos es menor en el último grupo, se ve que hay diferencias de un 5% en caso contrario de los civil y que son de un 95% los matrimonios con mayor diferencia.
```sh
pd.pivot_table(df, values= "anomat", index="clasemat", columns="total", aggfunc='count', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
```
```sh
pd.crosstab(df.clasemat, df.total,values=df.anomat,aggfunc=np.sum,normalize='columns').applymap(lambda x: "{0:.0f}%".format(100*x))
```
```sh
pd.crosstab(df.total, df.anomat,values=df.anomat,aggfunc=np.sum,normalize='columns').applymap(lambda x: "{0:.0f}%".format(100*x))
```
En este caso vemos por al atributo de ocupación, donde Profesionales científicos e intelectuales son de un 39% del grupo 0 de los 10 grupos.
```sh
df_grupo0 = df[df['total']==0]
clasemat_g0= df_grupo0[['total','grocuhom']].groupby('grocuhom').count().sort_values(['total'],ascending=False)
clasemat_g0['percentage'] = 100 * clasemat_g0['total']  / clasemat_g0['total'].sum()
clasemat_g0
```
```sh
df_grupo0 = df[df['total']==0]
df_grupo0
```
En este caso vemos por al atributo de ocupación, del grupo 1 vemos que los trabajadores de los servicios y vendedores de comercios y mercados son la mayor ocupación con un 21 % del grupo 1
```sh
df_grupo1 = df[df['total']==1]
clasemat_g1= df_grupo1[['total','grocuhom']].groupby('grocuhom').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
En caso de las mujeres se puede observar que en el área donde más se encuentra la mujer es como administradora del hogar con un 41%
```sh
df_grupo1 = df[df['total']==1]
clasemat_g1= df_grupo1[['total','grocumuj']].groupby('grocumuj').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
```sh
df_grupo1 = df[df['total']==1]
clasemat_g1= df_grupo1[['total','iuhom']].groupby('iuhom').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
También podemos observar que cantidad de hombres por urbanidad

```sh
df_grupo1 = df[df['total']==0]
clasemat_g1= df_grupo1[['total','iumuj']].groupby('iumuj').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
```sh
df_grupo1 = df[df['total']==0]
clasemat_g1= df_grupo1[['total','nachom']].groupby('nachom').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
También podemos observar que cantidad de mujeres por urbanidad
```sh
df_grupo1 = df[df['total']==0]
clasemat_g1= df_grupo1[['total','nacmuj']].groupby('nacmuj').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
Podemos observar que países son los presentados por parte de los hombres y Mujeres en el grupo 0 son los mismos países representados, para el grupo 1 sigue manteniéndose igual, los tres países representados.
```sh
df_grupo1 = df[df['total']==0]
clasemat_g1= df_grupo1[['total','nachom']].groupby('nachom').count().sort_values(['total'],ascending=False)
clasemat_g1['percentage'] = 100 * clasemat_g1['total']  / clasemat_g1['total'].sum()
clasemat_g1
```
#### Conclusiones:

- Según el análisis del dataset, se determina que las parejas que tienen menos diferencias entre sí, se casan por la iglesia católica y entre más son las diferencias entre pareja, se casan por lo civil.
- No se observan las diferencias exactas por edad, ya que, eran rangos muy diferentes que producían ruido. Esto lleva a poner un ponderado para establecer el rango de edad y verlo si era “true”, la edad del matrimonio era igual a 0 o igual a 5; caso contrario que enviará un “false” si era mayor a 5.
- La mejor forma para administrar los datos, es analizar los dataset por grupos. Esto contribuye a medir las distancias entre cada grupo y tener una visión más completa de la coincidencia que tienen los matrimonios. De ser diferentes, conocer las cantidades de matrimonios que representan dichas diferencias.
- Los datos muestran que casi un 41% de las mujeres en matrimonio son administradoras de hogar. Para ser 2020, es un porcentaje muy alto dado que actualmente las mujeres buscan ejercer una profesión o tener un trabajo fuera del hogar.
- Los datos analizados en el dataset, son matrimonios con nacionalidades de Costa Rica (la gran mayoría), Estados Unidos y Nicaragua. Esos tres países representan la mayor parte de los datos, sin embargo, existen otros países que también son parte de este listado.

#### Autores
En este proyecto final de la clase de Visualización de datos, por la Universidad LEAD fue elebarodao de la mano de:
- Dagoberto Herrera
- Natalia Herrrera
- su servidora Lolita Maldonado

