# Sistema de recomendación por filtrado colaborativo

Comparativa de **12 métodos de filtrado colaborativo** para predecir valoraciones de
restaurantes a partir de una matriz usuario × ítem muy dispersa, optimizando el
**MAE** (error absoluto medio).

> Práctica 2 de *Sistemes d'Informació i Organitzacions* — Grau en Enginyeria Informàtica, URV.
> Trabajo en pareja: Nawfal Aissaoui y Salma Jadiani.

## El problema

El dataset es una matriz de valoraciones usuario × restaurante en la que los huecos se
codifican con el valor centinela `99`. El primer paso es convertirlos a `NaN` para que
no contaminen medias ni similitudes. A partir de ahí, el reto es el habitual del
filtrado colaborativo: **la matriz es muy dispersa**, así que hay que estimar las
valoraciones ausentes sin sobreajustar a los pocos datos disponibles.

## Métodos evaluados

Los métodos se ordenan de menor a mayor sofisticación, y cada bloque corrige una
limitación del anterior:

**Imputación simple (líneas base)**
1. Media global
2. Media por usuario — captura que hay usuarios más generosos que otros
3. Media por ítem — captura que hay restaurantes mejores que otros
4. Baseline combinado — media global + sesgo de usuario + sesgo de ítem

**Vecindad (KNN)**

5. KNN user-based
6. KNN item-based
7. KNN item-based con normalización por usuario
8. KNN con **doble normalización**, en dos variantes: correlación de Pearson (k=10) y similitud del coseno (k=14)
9. KNN item-based con **shrinkage** y coseno / KNN con shrinkage por usuario (k=10)
10. KNN con doble shrinkage

La doble normalización resta a cada valoración tanto el sesgo del usuario como el del
ítem antes de calcular similitudes, de modo que la vecindad se calcula sobre
*desviaciones* y no sobre valores absolutos. El shrinkage penaliza las similitudes
calculadas con pocos co-ratings, que son las más ruidosas.

**Factorización de matrices**

11. SVD
12. SVD sobre la matriz doblemente normalizada

## Resultado

El mejor MAE lo obtuvo el **método 8: KNN item-based con doble normalización y
similitud del coseno (k=14)**. Su notebook y sus predicciones están aislados en
`mejor-resultado/`.

## Contenido

| Ruta | Descripción |
|---|---|
| `mejor-resultado/ScriptMillorMAE.ipynb` | Notebook del método ganador, listo para reejecutar |
| `mejor-resultado/resultats_knn_double_norm_cosine.csv` | Predicciones del método ganador |
| `experimentos/P2_SIO.ipynb` | Notebook completo con los 12 métodos y su evaluación |
| `experimentos/results/*.csv` | Predicciones de cada método, para comparar |
| `Memoria.pdf` | Memoria: metodología, tabla comparativa de MAE y conclusiones |

Los CSV de resultados llevan el formato `usuario;restaurante;predicción`, separados por `;`.

## Cómo ejecutar

Los notebooks se escribieron para **Google Colab** (usan `google.colab.files` para subir
el dataset). Para ejecutarlos en local basta con sustituir esa celda por una lectura
directa del fichero:

```python
df = pd.read_csv('recommendation_dataset.csv', sep=';', index_col=0)
```

Dependencias:

```bash
pip install pandas numpy scikit-learn matplotlib
```

> El dataset `recommendation_dataset.csv` lo proporcionaba la asignatura y no se
> incluye en el repositorio.

## Stack

Python · pandas · NumPy · scikit-learn · matplotlib · Jupyter / Colab
