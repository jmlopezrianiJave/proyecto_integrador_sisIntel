## Sistema inteligente de monitoreo mbiental y alertas tempranas

## Jose M. López, María C. Caicedo, Samuel F. Moncayo, Juan D. Valencia, Kevin Pérez

## 1. Declaración de uso de IA y reflexión ética

## 1.1. Uso de IA en el proyecto

En este proyecto, la inteligencia artificial se utilizó como herramienta de apoyo para orientar
el desarrollo técnico, estructurar ideas, mejorar la redacción, sugerir rutas de implementación y hacer refactoring.
No se usó para reemplazar el razonamiento. Por el contrario, las respuestas
obtenidas sirvieron como guía para organizar el flujo de trabajo, proponer estructuras iniciales de código,
identificar posibles errores metodológicos y mejorar la documentación del proyecto. Todo el código final fue
revisado, adaptado, depurado y validado manualmente por los integrantes del grupo antes de incorporarlo
al notebook principal.
En particular, la IA se usó como apoyo en tareas como la organización del EDA, la sugerencia de
estrategias para cruzar datos climáticos y eventos de desastre, la propuesta de estructuras para modelos
de clasificación y clustering, y la redacción de los apartados de análisis ético. El proyecto quedó final-
mente implementado con una estrategia de cruce multinivel y un pipeline de modelado adaptado a las
limitaciones reales de cobertura de datos.

## 1.2. Ejempls de prompts utilizados

1. Estructura general del proyecto
    “Estoy desarrollando un proyecto de IA sobre monitoreo ambiental y alertas tempranas con datos
    climáticos históricos de IDEAM y registros de desastres. Quiero una propuesta de estructura modu-
    lar en Python para organizar el flujo del proyecto en etapas: carga de datos, limpieza, EDA, cruce
    de variables climáticas con eventos, feature engineering, entrenamiento y evaluación. Necesito una
    guía clara de módulos y responsabilidades.”
2. Guía para EDA
    “Necesito ideas para realizar un análisis exploratorio de datos de temperatura, precipitación, hu-
    medad y eventos de desastre en un departamento de Colombia. Sugiere qué gráficos, métricas y
    preguntas analíticas conviene revisar para detectar patrones temporales, cobertura espacial, ano-
    malías y posibles relaciones entre clima y desastres.”
3. Limpieza y estandarización
    “Tengo datasets donde los nombres de municipios no coinciden exactamente entre fuentes. Dame
    una guía para estandarizar nombres, quitar tildes, unificar mayúsculas y preparar un cruce robusto
    entre datos climáticos y registros de eventos. Quiero entender la lógica antes de implementarla.”
4. Problema de cobertura espacial
    “En mi proyecto, muchos municipios con desastres no tienen estación climática propia. ¿Qué estra-
    tegias puedo usar para no perder esos casos al cruzar la información? Explícame alternativas como
    estación más cercana, agregación temporal o promedios regionales, con ventajas y riesgos de cada
    enfoque.”
5. Variables para modelos predictivos
    “Estoy construyendo un modelo para alertas tempranas por municipio y mes. ¿Qué features tendría
    sentido derivar a partir de temperatura, precipitación y humedad? Dame sugerencias explicando
    para qué serviría cada una.”
6. Evaluación adecuada
    “Mi problema está desbalanceado porque hay muchos más periodos sin desastre que con desas-
    tre. Oriéntame sobre qué métricas conviene usar en lugar de accuracy y cómo justificar F1-score,
    precision-recall, pesos de clase o técnicas como SMOTE en un contexto de alertas tempranas.”
7. Depuración conceptual del pipeline
    “Revisa esta idea de pipeline y dime qué errores metodológicos debería evitar: usar solo filas con
    desastre, hacer validación aleatoria en vez de temporal, entrenar con variables del futuro o evaluar
    con métricas poco adecuadas. No quiero que escribas todo el código; quiero que señales riesgos
    técnicos y cómo corregirlos.”
8. Mejora de redacción técnica
    “Voy a pegar párrafos de mi reporte técnico y necesito que me ayudes a mejorar claridad, coherencia
    y tono académico sin inventar resultados ni cambiar el contenido técnico. Solo quiero apoyo de
    redacción.”

## 1.3. Reflexión ética

El proyecto implementa un sistema inteligente de monitoreo ambiental y alertas tempranas que analiza
datos climáticos históricos y eventos de desastre para clasificar riesgo y detectar patrones en Santander.
El enunciado del curso exige que este tipo de agente no sea evaluado solo por su funcionamiento técnico,
sino también por su responsabilidad ética frente a decisiones, riesgos, sesgos e impacto humano.

1.3.1. Autonomía del sistema

El sistema tiene una autonomía parcial y controlada. No ejecuta acciones físicas ni ordena evacuaciones
por sí mismo, pero sí toma decisiones automáticas dentro del proceso analítico, como seleccionar reglas
de cruce de datos, calcular variables derivadas, estimar probabilidades de riesgo y clasificar eventos
potenciales. Por tanto, su autonomía es principalmente inferencial y computacional, mientras que la
decisión final sobre cómo actuar frente a una alerta debe permanecer en manos humanas.
En la práctica, el agente automatiza varias operaciones relevantes. Por ejemplo, prioriza el mejor
nivel de cruce disponible entre clima y desastre, usando primero datos del municipio, luego información
climática cercana y finalmente un promedio departamental cuando no existe cobertura local. También
ajusta umbrales de clasificación y combina evidencias del clasificador con variables de rareza territorial y
anomalía de precipitación dentro de un esquema bayesiano. Esto significa que, aunque no sea plenamente
autónomo en el despliegue operativo, sí influye de forma importante en qué situaciones aparecen como
riesgosas y cuáles no.
Desde una perspectiva ética, esto lo ubica como un agente de apoyo a decisiones. Esa clasificación es
consistente con la línea del proyecto definida en el curso, donde el agente analiza datos, clasifica riesgos
y detecta patrones, pero debe seguir siendo supervisado por humanos responsables. En consecuencia, la
autonomía del sistema no debe presentarse como absoluta, sino como una autonomía limitada por diseño.

1.3.2. Riesgos

Si el agente falla, las consecuencias pueden ser significativas porque el dominio de aplicación involucra
amenazas reales a comunidades humanas. El riesgo más grave corresponde a los falsos negativos, es decir,
casos en los que el sistema no emite una alerta pese a que sí existía una condición asociada a desastre. En
un contexto de incendios forestales, inundaciones, crecientes súbitas o movimientos en masa, una omisión
así podría traducirse en pérdida de tiempo de reacción, mayores daños materiales e incluso afectaciones
a la vida e integridad de las personas.
También existen riesgos por falsos positivos. Cuando el sistema alerta con demasiada frecuencia sin
que ocurra un evento relevante, las instituciones y comunidades pueden desarrollar fatiga de alerta y
perder confianza en la herramienta. En sistemas de prevención, este efecto es especialmente delicado,
porque una alerta verdadera futura puede ser ignorada si el historial previo estuvo lleno de alarmas poco
precisas.


Otro riesgo importante surge por las limitaciones de cobertura. Solo una fracción de los municipios
con desastres tiene estación climática propia, y una gran cantidad depende de promedios departamen-
tales como mecanismo de respaldo. Esto implica que en muchos casos el sistema opera con información
indirecta, menos precisa espacialmente, por lo que un error no necesariamente se debe a una mala técnica
de modelado, sino a una limitación estructural de los datos disponibles.
Por ello, el fallo del agente no solo debe entenderse como un error algorítmico, sino como un posible
error sociotécnico. Una autoridad podría confiar demasiado en la predicción y dejar de lado criterios
territoriales, conocimiento local o señales cualitativas que el modelo no captura. Éticamente, esto obliga
a tratar la salida del sistema como insumo para la decisión y no como sustituto del juicio profesional.

1.3.3. Sesgos

Los principales sesgos del sistema provienen de la forma en que los datos representan el territorio.
El proyecto evidencia que existen 94 municipios con desastres registrados, pero la cobertura climática
directa solo llega a una porción reducida de ellos, mientras que 69 municipios deben usar información
departamental como respaldo. Esto genera un sesgo espacial claro: el sistema conoce mejor los municipios
con infraestructura de medición y representa peor a los territorios con menor cobertura instrumental.
Además, los registros históricos de desastres no son una fotografía perfecta de la realidad, sino de lo
que fue reportado por las entidades responsables. Eso puede introducir sesgos de subregistro en zonas
rurales o con menor capacidad institucional, haciendo que el modelo aprenda patrones incompletos. En
otras palabras, un municipio puede parecer menos riesgoso no porque realmente lo sea, sino porque
históricamente ha sido menos observado o menos reportado.
El desbalance de clases también produce sesgo. El proyecto muestra que predominan los periodos sin
desastre sobre los periodos con desastre, lo cual empuja al sistema a favorecer predicciones negativas si
no se corrige adecuadamente. Por eso se justifican técnicas como pesos de clase, SMOTE y métricas como
F1-score o precision-recall en vez de accuracy. Sin estas precauciones, el modelo tendería a parecer bueno
simplemente por predecir casi siempre que no pasa nada.
A esto se suma un sesgo geográfico-climático derivado del uso de promedios regionales. En un depar-
tamento con gran variabilidad topográfica como Santander, usar el promedio departamental para inferir
condiciones locales puede borrar diferencias importantes entre municipios. Ese suavizado hace que el siste-
ma represente de forma desigual los contextos territoriales y, por tanto, produzca alertas potencialmente
menos precisas en zonas periféricas.

1.3.4. Impacto

El despliegue del sistema puede tener efectos positivos importantes si se usa de manera responsable.
Un sistema de alertas tempranas bien interpretado puede ayudar a priorizar vigilancia, orientar recursos
institucionales, reforzar monitoreo y anticipar escenarios de riesgo en municipios con antecedentes de de-
sastres. En ese sentido, puede mejorar la capacidad de respuesta de autoridades ambientales y organismos
de gestión del riesgo.
Sin embargo, el impacto no es uniforme para toda la población. Las comunidades ubicadas en muni-
cipios sin estación propia reciben una representación menos precisa de su realidad climática, por lo que
dependen de alertas construidas con información indirecta. Eso crea una desigualdad práctica: quienes
ya están en contextos con menor infraestructura podrían recibir un servicio algorítmico menos confiable.
También existe un impacto sobre los operadores humanos. Si el sistema no comunica claramente su
incertidumbre, el nivel de confianza de cada predicción y la calidad del dato usado en cada municipio, los
analistas pueden interpretar sus salidas como verdades categóricas. Esto favorece la sobreconfianza en la
automatización, un problema ético frecuente en sistemas de apoyo a decisiones.