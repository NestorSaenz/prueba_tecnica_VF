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

Para darle solución al proyecto, se verifica que las fuentes de los datos provienen de dos sitios diferentes. El primero consta de datos anonimizados de una empresa de telecomunicaciones en Bélgica, y el segundo conjunto de datos proviene de archivos en formato jsonl, los cuales contienen información de interacción de usuarios para un **e-commerce**.

