# IA embebida

# Resumen

La inteligencia artificial está dando pasos agigantados y para nosotros las personas enfocadas en hardware y en su control tenemos la ventaja de poder comprender cómo la IA funciona a bajo nivel, con posibilidad de comprenderla y realizar cambios o futuras optimizaciones a sistemas ya existentes. Para comprenderlo en arquitecturas ARM cortex-M podemos comenzar a comprender la unidad SMID la cual nos brinda un set de instrucciones en ensamblador para manejar cálculos de múltiples entradas y múltiples salidas, para ello se propone utilizar una Raspberry pi pico 2 la cual cuenta con este periférico, además de 2 núcleos de los cuales podríamos destinar uno para el cálculo de la RN y el segundo para interfaz o adecuación de datos, a su vez este cuenta con una FPU de precisión simple. Para utilizar estas instrucciones podríamos utilizarlas en ensamblador o a través del SDK oficial CMSIS-NN el cual nos dan una HAL capaz de manipular este ser de instrucciones en C.

# Objetivo 

Nuestro objetivo de centra en:

* Comprender como se diseña soluciones IA

* Comprender cómo interactúa la IA con el hardware 

* Comprender que limitaciones tiene el hardware 

* Implementar una pequeña red neuronal

# Importancia

En la actualidad tenemos varios microcontroladores que están empezando a tener estos periféricos enfocados a la IA como así también distintos entornos de desarrollo como por ejemplo de **NanoEdge AI Studio** o **Edge Impulse** que permiten diseño de IA embebido sin necesidad de profundizar en su funcionamiento.

# Porque IA en sistemas embebidos ?

La IA es una herramienta potente que nos permite enseñarle al hardware a reconocer patrones, en particular en sistemas embebidos podría utilizarse para mantenimiento predictivo, se instala un microcontrolador que sense un sistema y detecte cuando esté por ocurrir una falla.

Este sistema podría implementarse para CubeSats los cuales podrian encontrar fallas sin intervención humana, como así en numerosos sistemas críticos donde el humano no puede intervenir o en sistemas donde busquemos minimizar la intervención humana.

# Plan de acción 

## Fase 1: Investigación 

Propongo comenzar investigando hardware y herramientas utiles para el diseño e implementacion de IA con el objetivo de entender cómo es el flujo de trabajo de diseño de IA.

* Implementacion de librerias como TenserFlow y Keras 

## Fase 2: Práctica 

Comencemos por generar una pequeña red que implemente la operación lógica OR, está nos va a permitir entender cómo se comportará con distintos niveles de tensión en la entrada

* Implementacion en Python [``ptyhon/logic_or_rna/logic_or_model.py``]

* Implementacion en Python cuantizado en Q8 [**Pendiente**]

* Implementacion en CMSIS-NN [**Pendiente**]

## Fase 3: Desarrollo de RN convolucionales

Ampliemos la red neuronal a una convolucional, la cual nos permita por ejemplo clasificar señales cuadradas, senoidal es y triangulares, esto nos permitirá comprender cómo la IA puede reconocer patrones en el tiempo, la cual es útil para nosotros que manejamos señales tanto en el tiempo como en frecuencia 

* Implementacion en Python [**Pendiente**]

* Implementacion en Python cuantizado en Q8 [**Pendiente**]

* Implementacion en CMSIS-NN [**Pendiente**]

### Setup

El entrenamiento puede darse con una Raspberry Pi 5 la cual utilice un módulo generador de funciones y las generé aleatoriamente para obtener un dataset.

## Fase 4: Implementación 

Se podría implementar detección de fallas en sistemas mecánicos rotativos como por ejemplo un motor DC, ya sabemos cómo obtener vibraciones mecánicas podríamos extender el sistema de sensado añadiendo sensores de tensión y corriente. La IA se entrena con el objetivo de clasificar si el sistema está fallando o no, luego en el caso de falla que clasifique está falla.

* Implementacion en Python [**Pendiente**]

* Implementacion en Python cuantizado en Q8 [**Pendiente**]

* Implementacion en CMSIS-NN [**Pendiente**]

## Fase 5: Mejoras

Ya sabemos detectar patrones podríamos extender está capacidad con el uso del machine learning intentando predecir el comportamiento a futuro del sistema