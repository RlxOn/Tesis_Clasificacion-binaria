################## Bibliotecas necesarias #############
library(ggplot2)
library(dplyr)
library(andrews)
library(vcd)
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(corrplot)
library(psych)
library(coin)
library(nortest)
library(pROC)
library(car)
library(fmsb)
library(randomForest)
library(ggcorrplot)

################### Cargamos el datasets ###########
redwine <- read.csv("C:\\Users\\raulb\\Downloads\\winequality-red.csv",
                    header = T, sep = ";")

#Creamos una variable categorica con base al puntaje del vino
#1 si calidad >= 6 (buen vino), 0 e.o.c. (mal vino)
redwine <- redwine %>%
  mutate(Good.quality = ifelse(quality >= 6, 1,0))

redwine$Good.quality <- as.factor(redwine$Good.quality)

#Eliminamos la variable calidad ya que creamos la otra
redwine$quality <- NULL


################## Analisis de los datos###########
View(redwine)

##Resumen
summary(redwine)

##Tambien calculamos el CV y la sd de cada covariable explicativa
coefvar <- function(x){sd(x)/mean(x) * 100}
apply(redwine[1:11], 2, coefvar)
sd_vector <- apply(redwine[1:11], 2, sd) #Lo ocuparemos en el C. de Chauvenet
sd_vector

### Correlacion
ggcorrplot(cor(redwine[1:11]), method = "circle", hc.order = F,
           outline.color = "white", ggtheme = ggplot2::theme_bw,
           colors = c("#db7093","white","#70dbb8"))

### Distribucion
multi.hist(redwine[1:11], bcol = c("#cb3365", "#d14774","#db70c9"), dcol = NULL,
           global = FALSE)

### Valores atipicos y agrupamiento de datos

#Curva de Andrews
par(mfrow = c(1,1))
andrews(df = redwine, type = 1, clr = 3, step = 70, ymax = 4.1)

#Boxplots
par(mfrow = c(2,3))
for(i in 1:11){
boxplot(redwine[i], border = "#7b1f3d", col = "#eaadc1",main = names(redwine[i]))
}

par(mfrow = c(1,1))
  #Existen multiples valores atipicos en distintas variables
  #Ajustaremos el modelo con valores atipicos y posteriormente sin ellos

## Comparamos Medidas estadisticas de vinos de buena calidad y mala calidad
aggregate(. ~ Good.quality, data = redwine, median)
aggregate(. ~ Good.quality, data = redwine, mean)

# Si queremos aplicar la prueba t, aplicamos prueba de Anderson-Darling 
# A las variables explicativas
apply(redwine[1:11], 2 , FUN = ad.test)
  
  #En todas las variables se rechaza normalidad pues p-value < 0.05
  #Recurrimos a la prueba no parametrica de Wilcoxon

## Prueba de Mann-Withney- Wilxcoxon

#Ho: La diferencia de medias es igual a cero (m1 = m2)
#H1: La diferencia de medias es distinta de cero (m1 != m2)

wilcox_test(fixed.acidity ~ Good.quality, data = redwine)
wilcox_test(volatile.acidity ~ Good.quality, data = redwine) 
wilcox_test(citric.acid ~ Good.quality, data = redwine)
wilcox_test(residual.sugar ~ Good.quality, data = redwine) #No rechazo H0
wilcox_test(chlorides ~ Good.quality, data = redwine)
wilcox_test(free.sulfur.dioxide ~ Good.quality, data = redwine)
wilcox_test(total.sulfur.dioxide ~ Good.quality, data = redwine)
wilcox_test(density ~ Good.quality, data = redwine)
wilcox_test(pH ~ Good.quality, data = redwine) #No rechazo H0
wilcox_test(sulphates ~ Good.quality, data = redwine)
wilcox_test(alcohol ~ Good.quality, data = redwine)
      #Solo el azucar residual y el pH son estadisticamente iguales

############ Particion para el conjunto para prueba-entrenamiento #############
n <- dim(redwine)[1]
set.seed(1989)  ##Taylor's version
entrenamiento <- sample(1:n, 0.66*n)    #2/3 para entrenamiento

redwine.prueba <- redwine[-entrenamiento,]
redwine.entrenamiento <- redwine[entrenamiento,]

yentrenamiento <- redwine$Good.quality[entrenamiento]
yprueba <- redwine$Good.quality[-entrenamiento]


######################### Modelo 1 de RG ######################################
log_model01 <- glm(Good.quality ~ ., data = redwine.entrenamiento, family = binomial)
summary(log_model01)
confint(log_model01)
vif(log_model01)
step(log_model01)
  #Existe un problema de multicolinealidad debido a que 5 covariables
  #No son estadisticamente significativas, ademas de tener intervalos de 
  #Confianza (al 95%) muy amplios y VIF mayores a 5
  
  #Eliminamos Acidez fija, azucar residual, densidad, pH, acido citrico

## Analisis del ajuste
devianza01 <- deviance(log_model01)
AIC01 <- AIC(log_model01)

# Matriz de confusion y Curva ROC
pred01 <- predict.glm(log_model01, newdata = redwine.prueba, type = "response")

objroc01 <- pROC::roc(yprueba, pred01,auc=T,ci=T) #respuesta, predictor
plot.roc(objroc01,print.auc=T,print.thres = "best",
         col="#db7093", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

pred01 <- as.factor(ifelse(pred01 >= 0.6,1,0))    #Guardamos en levels 0,1 para la matriz 
confusionMatrix(data = pred01, redwine.prueba$Good.quality)
Prec01 <- 0.7169
IC01 <- "(0.677,0.7544)"
Sens01 <- 0.8287
Esp01 <- 0.6212 
AUC01 <- objroc01$auc[1]


########################## Ajuste del segundo MRL #############################
log_model02 <- glm(Good.quality ~ volatile.acidity + chlorides + 
                    free.sulfur.dioxide + total.sulfur.dioxide + sulphates + 
                    alcohol, data = redwine.entrenamiento, family = binomial)

summary(log_model02)
confint(log_model02)
vif(log_model02)
step(log_model02)
    #No hay multicolinealidad en nuestras variables

## Analisis del modelo 2
devianza02 <- deviance(log_model02)
AIC02 <- AIC(log_model02)

##Matriz de confusion y ROC
pred02 <- predict.glm(log_model02, newdata = redwine.prueba, type = "response")
objroc02 <- pROC::roc(yprueba, pred02,auc=T,ci=T)
plot.roc(objroc02,print.auc=T,print.thres = "best",
         col="#db7093", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

pred02 <- as.factor(ifelse(pred02 >= 0.6,1,0))
confusionMatrix(data = pred02, redwine.prueba$Good.quality)
Prec02 <- 0.7169
IC02 <- "(0.677,0.7544)"
Sens02 <- 0.8287
Esp02 <- 0.6212
AUC02 <- objroc02$auc[1]

############################## Valores atipicos ###############################

#Guardamos un data frame al cual le quitaremos valores atipicos
#Aplicaremos el criterio de Chauvenet 

#Calculamos el coeficiente Kn

Qn <- 1/(4*dim(redwine)[1]) #Cuantil para la normal
Kn <- qnorm(Qn, mean = 0, sd = 1, lower.tail = FALSE)
mean_vector <- apply(redwine[1:11], 2, mean) #Vector de medias
#Como queremos los datos |Xi - Xbarra | > KnS
#Xi debe estar en el intervalo (Xbarra - KnS, Xbarra + KnS)

redwine_outliers <- redwine %>%
  filter(fixed.acidity < mean_vector[1]+ Kn * sd_vector[1]) %>%
  filter(volatile.acidity < mean_vector[2] + Kn * sd_vector[2]) %>%
  filter(citric.acid < mean_vector[3] + Kn * sd_vector[3]) %>%
  filter(residual.sugar < mean_vector[4] + Kn * sd_vector[4]) %>%
  filter(chlorides < mean_vector[5] + Kn * sd_vector[5]) %>%
  filter(free.sulfur.dioxide < mean_vector[6] + Kn * sd_vector[6]) %>%
  filter(total.sulfur.dioxide < mean_vector[7] + Kn * sd_vector[7]) %>%
  filter(density < mean_vector[8] + Kn * sd_vector[8]) %>%
  filter(pH < mean_vector[9] + Kn * sd_vector[9]) %>%
  filter(sulphates < mean_vector[10] + Kn * sd_vector[10]) %>%
  filter(alcohol < mean_vector[11] + Kn * sd_vector[11])

##Hacemos un resumen
summary(redwine_outliers)    #Quedaron 1520 vinos (el 95%)

##Analisis de medidas
aggregate(. ~ Good.quality, data = redwine_outliers, mean)
aggregate(. ~ Good.quality, data = redwine_outliers, median)

##Curva de andrews
par(mfrow = c(1,1))
andrews(redwine_outliers, type = 1, clr = 3, step = 60, ymax = 5) 

##Boxplots
par(mfrow = c(2,3))
for(i in 1:11){
  boxplot(redwine_outliers[i], border = "#7b1f3d", col = "#e599b2",
          main = names(redwine_outliers[i]))
}

##Histogramas
par(mfrow = c(1,1))
multi.hist(redwine_outliers[1:11], bcol = c("#70c9db", "#70dbb8","#93db70"), 
           dcol = NULL, global = FALSE)

#################### particion de data 2 (sin valores atipicos) ###########
n <- dim(redwine_outliers)[1]
entrenamiento2 <- sample(1:n, 0.66*n)

redwine.prueba2 <- redwine_outliers[-entrenamiento2,]
redwine.entrenamiento2 <- redwine_outliers[entrenamiento2,]

yentrenamiento2 <- redwine_outliers$Good.quality[entrenamiento2]
yprueba2 <- redwine_outliers$Good.quality[-entrenamiento2]


############# Modelo 3 (no datos atipicos)################
log_model03 <- glm(Good.quality ~ .,
                  data = redwine.entrenamiento2, family = binomial)
summary(log_model03)
confint(log_model03)
vif(log_model03)
step(log_model03)
      #Existe multicolinealidad
      #QUitamos Acidez fija, acido citrico, azucar residual, cloruros
      #SO2 libre, densidad y pH.

##Analisis del modelo 3
devianza03 <- deviance(log_model03)
AIC03 <- AIC(log_model03)

pred03 <- predict.glm(log_model03, newdata = redwine.prueba2, type = "response")
objroc03 <- pROC::roc(yprueba2, pred03,auc=T,ci=T)
plot.roc(objroc03,print.auc=T,print.thres = "best",
         col="#33cb9a", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

pred03 <- as.factor(ifelse(pred03 >= 0.6,1,0))
confusionMatrix(data = pred03, redwine.prueba2$Good.quality)

Prec03 <- 0.7524
IC03 <- "(0.7129,0.7891)"
Sens03 <- 0.8167
Esp03 <- 0.6917
AUC03 <- objroc03$auc[1]

################ Modelo 4 (outliers 2)############
log_model04 <- glm(Good.quality ~ volatile.acidity + total.sulfur.dioxide + 
                  sulphates + alcohol, data = redwine.entrenamiento2, 
                  family = binomial)
summary(log_model04)
confint(log_model04)
vif(log_model04)
step(log_model04)
    #Todos los coeficientes son significativos y no hay multicolinealidad

##Analisis del modelo 04
devianza04 <- deviance(log_model04)
AIC04 <- AIC(log_model04)

pred04 <- predict.glm(log_model04, newdata = redwine.prueba2, type = "response")
objroc04 <- pROC::roc(yprueba2, pred04,auc=T,ci=T)
plot.roc(objroc04,print.auc=T,print.thres = "best",
         col="#33cb9a", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

pred04 <- as.factor(ifelse(pred04 >= 0.6,1,0))
confusionMatrix(data = pred04, redwine.prueba2$Good.quality)

Prec04 <- 0.7485
IC04 <- "(0.7088, 0.7854)"
Sens04 <- 0.8088
Esp04 <- 0.6917
AUC04 <- objroc04$auc[1]

################ Comparacion de los 4 modelos ####################
Modelos <- c("Modelo 1" , "Modelo 2", "Modelo 3", "Modelo 4")
Precision <- c(Prec01, Prec02, Prec03, Prec04)
IC95 <- c(IC01, IC02, IC03, IC04)
Sensibilidad <- c(Sens01, Sens02, Sens03, Sens04)
Especificidad <- c(Esp01, Esp02, Esp03, Esp04)
AIC <- c(AIC01, AIC02, AIC03, AIC04)
Devianza <- c(devianza01, devianza02, devianza03, devianza04)
AUC <- c(AUC01, AUC02, AUC03, AUC04)

Log_comparacion <- data.frame(Modelos, Precision, IC95, Sensibilidad,
                              Especificidad, AIC, Devianza, AUC)

View(Log_comparacion)

#Visualizamos los residuales de los modelos
par(mfrow = c(2,2))
hist(log_model01$residuals, probability = TRUE, col = "#db7093",
     main = "Residuales Modelo 1")

hist(log_model02$residuals, probability = TRUE, col = "#db7093",
     main = "Residuales Modelo 2")

hist(log_model03$residuals, probability = TRUE, col = "#93db70",
     main = "Residuales Modelo 3")

hist(log_model04$residuals, probability = TRUE, col = "#93db70",
     main = "Residuales Modelo 4")
    
    #Si nos fijamos unicamente en las metricas, elegimos el modelo 03
    #Si tambien verificamos la multicolinealidad, elegimos el modelo 04

################ Clasificador de Bayes ####################

##### Clasificador con el conjunto original
Bayes1 <- naiveBayes(Good.quality ~ ., data = redwine.entrenamiento)
bayes_pred01 <- predict(Bayes1, newdata = redwine.prueba)

confusionMatrix(data = bayes_pred01, redwine.prueba$Good.quality)

bayes_pred01 <- as.numeric(bayes_pred01)   #Lo guardamos numerico para el objeto roc
bayes_objroc01 <- pROC::roc(yprueba, bayes_pred01,auc=T,ci=T)
par(mfrow = c(1,1))
plot.roc(bayes_objroc01,print.auc=T,print.thres = "best",
         col="#db8370", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))
bayes_objroc01$auc[1]


#### Clasificador para datos atipicos
Bayes2 <- naiveBayes(Good.quality ~ ., data = redwine.entrenamiento2)
bayes_pred02 <- predict(Bayes2, newdata = redwine.prueba2)

confusionMatrix(data = bayes_pred02, redwine.prueba2$Good.quality)

bayes_pred02 <- as.numeric(bayes_pred02)
bayes_objroc02 <- pROC::roc(yprueba2, bayes_pred02,auc=T,ci=T)
plot.roc(bayes_objroc02,print.auc=T,print.thres = "best",
         col="#db8370", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))
bayes_objroc02$auc[1]

##################### Maquinas de soporte vectorial ###################

#### Conjunto con datos atipicos
##Kernel lineal
  #Utilizamos la funcion tune para ver el mejor valor del parametro C
svm_tune_lin <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento,
                     kernel = "linear", 
                     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 1.5,2,5, 10, 20)))

svm_tune_lin$best.parameters    #el mejor parametro es costo = 0.1

svm_lineal01 <- svm(Good.quality ~ ., data = redwine.entrenamiento, kernel = "linear",
             cost = 0.1)

#Hacemos las predicciones con el conjunto de prueba
svm_lineal01_pred <- predict(svm_lineal01, redwine.prueba)
confusionMatrix(data = svm_lineal01_pred, redwine.prueba$Good.quality)

svm_lineal01_roc <- pROC::roc(yprueba, as.numeric(svm_lineal01_pred),auc=T,ci=T)

plot.roc(svm_lineal01_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_lineal01_roc$auc[1]   #0.7280299

## Kernel gaussiano
svm_tune_gauss <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento,
                       kernel = "radial", 
                       ranges = list(cost = c(0.1,0.5,1,1.5,5,10,15),
                                     gamma = c(0.1,0.5,1,1.5,2,5,10)))
svm_tune_gauss$best.parameters    #Elegimos costo = 5, gamma = 0.5

svm_gauss01 <- svm(Good.quality ~ ., data = redwine.entrenamiento, kernel = "radial",
                    cost = 5, gamma = 0.5)

svm_gauss01_pred <- predict(svm_gauss01, redwine.prueba)
confusionMatrix(data = svm_gauss01_pred, redwine.prueba$Good.quality)

svm_gauss01_roc <- pROC::roc(yprueba, as.numeric(svm_gauss01_pred),auc=T,ci=T)

plot.roc(svm_gauss01_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_gauss01_roc$auc[1]  # 0.7228968

##Kernel polinomial
svm_tune_poly <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento2,
                       kernel = "polynomial", 
                       ranges = list(cost = c(0.1,0.5,1,1.5),
                                     gamma = c(0.1,0.5,1,1.5),
                                     coef0 = c(0.1,0.5,1,1.2),
                                     degree = 2))
svm_tune_poly$best.model

svm_poly02 <- svm(Good.quality ~ .,data = redwine.entrenamiento2, kernel = "polynomial",
                  cost = 1, degree = 2, coef0 = 0.5)

svm_poly02_pred <- predict(svm_poly02, redwine.prueba2)
confusionMatrix(data = svm_poly02_pred, redwine.prueba2$Good.quality)

svm_poly02_roc <- pROC::roc(yprueba2, as.numeric(svm_poly02_pred),auc=T,ci=T)

plot.roc(svm_poly02_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_poly02_roc$auc[1]    #0.7126715


## kernel sigmoidal
svm_tune_sigm <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento,
                      kernel = "sigmoid",
                      ranges = list(cost = c(0.1,0.2,1,5),
                                    gamma = c(0.1,0.2,1),
                                    coef0 = c(0.1,0.2,1)))
svm_tune_sigm$best.parameters    #costo = 0.1, gamma = 0.1, coef0 = 0.1

svm_sigm01 <- svm(Good.quality ~ .,data = redwine.entrenamiento, kernel = "sigmoid",
                  cost = 0.1, gamma = 0.1, coef0 = 0.1)

svm_sigm01_pred <- predict(svm_sigm01, redwine.prueba)
confusionMatrix(data = svm_sigm01_pred, redwine.prueba$Good.quality)

svm_sigm01_roc <- pROC::roc(yprueba, as.numeric(svm_sigm01_pred),auc=T,ci=T)

plot.roc(svm_sigm01_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_sigm01_roc$auc[1]    #0.7126715

#### Conjunto sin datos atipicos
##Kernel lineal
svm_tune_lin <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento2,
                     kernel = "linear", 
                     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 1.5,2,2.1,2.2,5, 10, 20)))

summary(svm_tune_lin)
svm_tune_lin$best.parameters    #el mejor parametro es costo = 2.0

svm_lineal02 <- svm(Good.quality ~ ., data = redwine.entrenamiento2, kernel = "linear",
                    cost = 1)

#Hacemos las predicciones con el conjunto de prueba
svm_lineal02_pred <- predict(svm_lineal02, redwine.prueba2)
confusionMatrix(data = svm_lineal02_pred, redwine.prueba2$Good.quality)

svm_lineal02_roc <- pROC::roc(yprueba2, as.numeric(svm_lineal02_pred),auc=T,ci=T)

plot.roc(svm_lineal02_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_lineal02_roc$auc[1]   #0.70954482

## Kernel gaussiano
svm_tune_gauss <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento2,
                       kernel = "radial", 
                       ranges = list(cost = c(0.1,0.5,1,1.5,5,10,15),
                                     gamma = c(0.1,0.5,1,1.5,2,5,10)))
svm_tune_gauss$best.parameters    #Elegimos costo = 5, gamma = 0.1

svm_gauss02 <- svm(Good.quality ~ ., data = redwine.entrenamiento2, kernel = "radial",
                   cost = 5, gamma = 0.1)

svm_gauss02_pred <- predict(svm_gauss02, redwine.prueba2)
confusionMatrix(data = svm_gauss02_pred, redwine.prueba2$Good.quality)

svm_gauss02_roc <- pROC::roc(yprueba2, as.numeric(svm_gauss02_pred),auc=T,ci=T)

plot.roc(svm_gauss02_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_gauss02_roc$auc[1]  # 0.773975

##Kernel polinomial
svm_tune_poly <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento,
                      kernel = "polynomial", 
                      ranges = list(cost = c(0.1,0.5,1,1.5),
                                    gamma = c(0.1,0.5,1,1.5),
                                    coef0 = c(0.1,0.5,1,1.2),
                                    degree = 3))
svm_tune_poly$best.performance
svm_tune_poly$best.model

svm_poly01 <- svm(Good.quality ~ .,data = redwine.entrenamiento, kernel = "polynomial",
                  cost = 0.5, degree = 2, coef0 = 1)

summary(svm_poly01)

svm_poly01_pred <- predict(svm_poly01, redwine.prueba)
confusionMatrix(data = svm_poly01_pred, redwine.prueba$Good.quality)

svm_poly01_roc <- pROC::roc(yprueba, as.numeric(svm_sigm01_pred),auc=T,ci=T)

plot.roc(svm_poly01_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_poly01_roc$auc[1]    #0.7126715


## kernel sigmoidal
svm_tune_sigm <- tune("svm", Good.quality ~ ., data = redwine.entrenamiento2,
                      kernel = "sigmoid",
                      ranges = list(cost = c(0.1,0.2,1,5),
                                    gamma = c(0.1,0.2,1),
                                    coef0 = c(0.1,0.2,1)))
svm_tune_sigm$best.parameters    #costo = 0.1, gamma = 0.1, coef0 = 0.1

svm_sigm02 <- svm(Good.quality ~ .,data = redwine.entrenamiento2, kernel = "sigmoid",
                  cost = 0.1, gamma = 0.1, coef0 = 0.1)

svm_sigm02_pred <- predict(svm_sigm02, redwine.prueba2)
confusionMatrix(data = svm_sigm02_pred, redwine.prueba2$Good.quality)

svm_sigm02_roc <- pROC::roc(yprueba2, as.numeric(svm_sigm02_pred),auc=T,ci=T)

plot.roc(svm_sigm02_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

svm_sigm02_roc$auc[1]    #0.7058046

############################ Arboles ##########################################
#Creamos el arbol con RPART
#Como y es categorica, utilizamos method class
arbol01 <- rpart(Good.quality ~ .,data = redwine.entrenamiento, method = "class")

#Dibujamos el arbol de decision
par(mfrow = c(1,1))
rpart.plot(arbol01,type = 1, extra = 2,under = T,varlen = 0,faclen = 0,
           fallen.leaves = TRUE, space = 2,gap = 0,tweak = 1.5)

arbol01_pred <- predict(arbol01, newdata = redwine.prueba, type = "class")
confusionMatrix(arbol01_pred,yprueba)
  #Primer arbol ajustado automaticamente, sin afinar el CP


#Realizamos validacion cruzada para afinar el CP con un nuevo arbol 
arbol02_train <- train(Good.quality ~ ., data = redwine.entrenamiento, method = "rpart",
                trControl = trainControl(method = "cv", number = 20),
                tuneLength = 5) #CV por cross-validation

plot(arbol02_train)   #Grafica el CP vs la precision
arbol02_train$bestTune    #El mejor CP = 0.01014199
mean(arbol02_train$resample$Accuracy)   #Media de la precision

arbol02 <- arbol02_train$finalModel    #Guardamos el mejor modelo
summary(arbol02)

#Dibujamos el arbol con el mejor CP y la importancia de las variables
rpart.plot(arbol02, extra = 2, under = TRUE,  varlen = 0, faclen = 0,
           fallen.leaves = TRUE, space = 2, tweak = 1.5)

dotPlot(varImp(arbol01_ajuste, compete=FALSE))
    #Las variables mas importantes son Alcohol, sulfitos, Acidez y cloruros


#Aplicamos los datos de prueba
arbol02_pred <- predict(arbol02, newdata = redwine.prueba, type = "class")
confusionMatrix(arbol02_pred,yprueba)

arbol02_roc <- pROC::roc(yprueba, as.numeric(arbol02_pred),auc=T,ci=T)

plot.roc(arbol02_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

arbol02_roc$auc[1]    #0.7089798


## Segundo arbol (conjunto sin valores atipicos)
arbol03_train <- train(Good.quality ~ ., data = redwine.entrenamiento2, method = "rpart",
                       trControl = trainControl(method = "cv", number = 20),
                       tuneLength = 5)
plot(arbol03_train)
arbol03_train$bestTune    #El mejor CP = 0.01287554
mean(arbol03_train$resample$Accuracy)
arbol03 <- arbol03_train$finalModel

#Dibujamos el arbol 3
rpart.plot(arbol03, extra = 2, under = TRUE,  varlen = 0, faclen = 0,
           fallen.leaves = TRUE, space = 2, tweak = 1.5)

dotPlot(varImp(arbol01_ajuste, compete=FALSE))
#Las variables mas importantes son Alcohol, sulfitos, Acidez y cloruros


#Aplicamos los datos de prueba
arbol03_pred <- predict(arbol03, newdata = redwine.prueba2, type = "class")
confusionMatrix(arbol03_pred,yprueba2)

arbol03_roc <- pROC::roc(yprueba2, as.numeric(arbol03_pred),auc=T,ci=T)

plot.roc(arbol03_roc,print.auc=T,print.thres = "best",
         col="#db70c9", lwd =3,xlab="1-Especificidad",ylab="Sensibilidad", grid = TRUE,
         xlim = c(1.0,-0.3))

arbol03_roc$auc[1]    #0.6930424


############################BOSQUES ALEATORIOS#################################
#Usamos train para encontrar los mejores parametros del bosque
randomFit <- train(Good.quality ~ .,  data = redwine.entrenamiento, method = "rf",
                   tuneLength = 10,
                   trControl = trainControl(
                     method = "cv", number = 20))
bosque01 <- randomFit$finalModel
plot(bosque01) #Num de arboles vs Error out of the bag
varImpPlot(bosque01, main = "Importancia en Bosque 01", color = c("#b72e5b","#4774d1"),
           lcolor = "#47d1a4", cex = 1.2)

error_frame <- as.data.frame(bosque01$err.rate)
which.min(error_frame$OOB) #Un bosque con 101 arboles minimiza el OOB
plot(bosque01)

bosque01_pred <- predict(bosque01, newdata = redwine.prueba, type = "class")
confusionMatrix(bosque01_pred, yprueba)


#Construimos un bosque con 101 arboles y mtry = 6
bosque02 <- randomForest(Good.quality ~., data = redwine.entrenamiento,
                         ntree = 101, mtry = 6)

varImpPlot(bosque02, main = "Importancia de los predictores", color = c("#b72e5b","#4774d1"),
           lcolor = "#47d1a4", cex = 1.2)

bosque02_pred <- predict(bosque02, newdata = redwine.prueba, type = "class")
confusionMatrix(bosque02_pred, yprueba)

##Bosque sin datos atipicos
randomFit <- train(Good.quality ~ .,  data = redwine.entrenamiento2, method = "rf",
                   tuneLength = 10,
                   trControl = trainControl(
                     method = "cv", number = 20))
randomFit$finalModel #mtry = 3, ntrees = 63
error_frame <- as.data.frame(randomFit$finalModel$err.rate)
which.min(error_frame$OOB)

bosque03 <- randomForest(Good.quality ~., data = redwine.entrenamiento2,
                         ntree = 63, mtry = 3)

varImpPlot(bosque03, main = "Importancia de los predictores", color = c("#b72e5b","#4774d1"),
           lcolor = "#47d1a4", cex = 1.2)

bosque03_pred <- predict(bosque03, newdata = redwine.prueba2, type = "class")
confusionMatrix(bosque03_pred, yprueba2)
