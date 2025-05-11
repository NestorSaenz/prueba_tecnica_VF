# Sistema de Segmentación y Mensajería Personalizada

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange.svg)
![Azure-OpenAI](https://img.shields.io/badge/Azure%20OpenAI-API-lightgrey)

Solución técnica para segmentación de miembros y generación de mensajes personalizados basados en comportamiento de churn.

## 📌 Descripción

Este proyecto implementa:
1. **Segmentación de miembros** mediante análisis PCA y clustering
2. **Generación de mensajes personalizados** usando Azure OpenAI
3. Integración con datos anonimizados del sector telecomunicaciones

## 🛠️ Estructura del Proyecto

## 📊 EDA:

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



La mayoria de los eventos son interacciones a través de clics .


