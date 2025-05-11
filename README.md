# Sistema de Segmentación y Mensajería Personalizada

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Azure-OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-lightgrey)

Solución técnica para segmentación de miembros y generación de mensajes personalizados basados en comportamiento de churn.

# 📌 Descripción

Este proyecto implementa:
1. **Segmentación de miembros** mediante análisis PCA y clustering
2. **Generación de mensajes personalizados** usando Azure OpenAI
3. Integración con datos anonimizados del sector telecomunicaciones

## 🛠️ Estructura del Proyecto

![Estructura](https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/estructura.jpg)


# 📊 EDA

Para darle solución al proyecto, se verifica que las fuentes de los datos provienen de dos sitios diferentes. El primero consta de datos anonimizados de una empresa de telecomunicaciones en Bélgica, y el segundo conjunto de datos proviene de archivos en formato jsonl, los cuales contienen información de interacción de usuarios para un *e-commerce*.

1. **Churn**
para elaborar el EDA se llevo a acbo en visual Studio Code en un archivo .ipynb, evidenciando que se encuentran 11896 registros y 180 columnas los cuales estan anonimizados por cuestion de seguridad, y como target se encuentra la columna (y) la cual muestra si hay churn o no para un usuario, se obtiene que la probabilidad de churn del conjunto de datos es del 3.43%, y se le hizo un tratamiento a los clientes es decir comunicación con ellos al 75.74%.

A continuación, se visualiza una grafico de barras con las correlaciones absolutas de las variables con la columna objetivo o traget

![Correlación de variables](https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/correlacion.png)

Matriz de correlación de variables

![Matriz de Correlación de variables](https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/matriz.png)

No se observa un fuerte correlación entre las variables

2. **OTTO**

Este conjunto de datos esta en formato jsonl, es bastante pesado, para iniciar con su análisis se cargo en el ambiente por lotes con un chun_size=10000, esta dividido en train y test en conjunto aprozimadamente 230M de registros con 5 columnas. 
**-Duplicados y valores faltantes**
No se observan valores faltantes, sin embargo hay varios duplicados, los cuales se proceden a borrarse
se analiza la distribución de eventos tal como se muestra a continuación:

## 🔍 Hallazgos Clave

Visualmente se observa el análisis temporal y los articulos mas populares
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/raw/main/imagenes/distribucion_eventos.jpg" width="95%" alt="Distribución de Eventos">
      <br><em>Figura 1: Distribución general de eventos</em>
    </td>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/raw/main/imagenes/grafica_distribucion_eventos.png" width="95%" alt="Gráfica de Distribución">
      <br><em>Figura 2: Detalle de distribución por categoría</em>
    </td>
  </tr>
</table>



La mayoria de los eventos son interacciones a través de clics.

## 📈 Dinámica de Interacciones

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/raw/main/imagenes/Analisis_temporal.png" width="100%" alt="Análisis Temporal">
      <br>
      <strong>Patrones Temporales</strong><br>
      <em>Figura 3</em>: Distribución horaria/semanal de eventos<br>
      <small></small>
    </td>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/raw/main/imagenes/articulos_mas_interactuados.png" width="100%" alt="Top Artículos">
      <br>
      <strong>Top Contenidos</strong><br>
      <em>Figura 4</em>: Artículos con mayor interacción<br>
      <small></small>
    </td>
  </tr>
</table>

# 🤖 Modelado
## 1. **Segmentación de usuarios**
una vez entendido los datos se realiza lo que es la segmentacion de los clientes, para ello el set de datos se reduce a 50 componentes principales aplicando PCA

🎯 Objetivos
- Calcular puntuaciones de engagement de usuarios
- Segmentar clientes según su comportamiento
- Identificar patrones clave en interacciones
- Crear segmentos de clientes accionables
  
 ### Análisis de Componentes Principales

#### PCA aplicado para reducir dimensionalidad
Ponderación de componentes:
- PC_0: 50% (Más importante)
- PC_1: 30%
- PC_2: 15%
- PC_3: 5%

#### Caracteristicas analizadas
- Interacciones de usuarios
- Frecuencia de sesiones
- Patrones de actividad
- Métricas de uso

Para hacer la segmentación se utilizó Los vecinos mas cercanos KMEANS, obteniendo los siguientes resultados:

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/segmentacion.jpg" width="100%" alt="Resultados Segmentación">
      <br>
      <strong>Patrones Temporales</strong><br>
      <em>Figura 3</em>: Resultados Segmentación<br>
      <small></small>
    </td>
    <td align="center">
      <img src="https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/grafico_segmentacion.png" width="100%" alt="Gráfico de la Segmentación">
      <br>
      <strong>Top Contenidos</strong><br>
      <em>Figura 4</em>: Gráfico de la Segmentación<br>
      <small></small>
    </td>
  </tr>
</table>
con la siguiente métrica de evaluación AUC-ROC: 0.7108018085367853

En el siguiente grafico se observa la distribución de la segmentación

![Distribución de la Segmentación](https://github.com/NestorSaenz/prueba_tecnica_VF/blob/main/imagenes/grafica_segmentacion.png)

## 2. **Mensajes personalizados**
En este apartadon se implementa un sistema de generación automática de mensajes personalizados para clientes, utilizando Azure OpenAI y datos de segmentación de usuarios.

🔄 Proceso
1. Configuración Inicial
- Inicialización del cliente Azure OpenAI
- Configuración de credenciales y endpoint(por temas de tiempo deje las credenciales hardcodeadas en el código, pero van en las variables de entorno)
- Importación de librerías necesarias

2. Carga de datos
   
- Carga de datos segmentados de clientes
- Incluye métricas de engagement y probabilidad de abandono

3. Generación de Mensajes
   
Sistema de generación basado en:

Grupos de Riesgo:

1: Cliente fiel (bajo riesgo)

2: Cliente con señales de riesgo

3: Cliente que requiere atención inmediata

4. Estructura del Mensaje
Cada mensaje personalizado incluye:

- Nivel de engagement del cliente
- Probabilidad de abandono
- Recomendaciones específicas de productos VoiceFlip
- Firma profesional
- Llamada a la acción
  
5. Características del Prompt
   
-Formal y directo

-Métricas específicas de uso

En el siguiente párrafo se puede ver un ejemplo de un mensaje personalizado para una persona segmentada como de riesgo medio:
"Estimado/a [Nombre del Cliente],

Espero que este mensaje le encuentre bien. Nos complace observar que su nivel actual de engagement con VoiceFlip es de 5.35, lo cual refleja un uso constante y significativo de nuestra plataforma. Además, su probabilidad de abandono, calculada en un 0.05, confirma una relación estable con nuestros servicios. ¡Gracias por confiar en nosotros!

Para continuar optimizando su experiencia y el valor que obtiene de VoiceFlip, me gustaría recomendarle dos de nuestras soluciones más avanzadas que pueden complementar su flujo de trabajo actual:

1. **VoiceFlip Insights Pro**: Esta herramienta avanzada de análisis le permitirá profundizar aún más en los patrones de uso de voz y audio, proporcionando métricas predictivas que pueden mejorar la toma de decisiones en tiempo real.

2. **VoiceFlip API Connect**: Integrar esta solución a su sistema actual le permitirá personalizar aún más el procesamiento de voz, adaptándolo a las necesidades específicas de su negocio. Es ideal para empresas que buscan flexibilidad y escalabilidad.

Ambas soluciones están diseñadas para maximizar los beneficios de nuestra plataforma y habilitar su crecimiento continuo. Me encantaría coordinar una breve llamada o demostración personalizada para explorar cómo estas herramientas pueden adaptarse a sus objetivos.

Por favor, no dude en responder a este correo o contactarme directamente al [número de teléfono] para agendar una reunión. Estamos aquí para apoyarle en cada paso del camino.

Atentamente,  
[Nombre y Apellido]  
Especialista en Retención de Clientes  
VoiceFlip  "



