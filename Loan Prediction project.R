if(!require(dplyr)) {install.packages("dplyr"); require(dplyr)}
if(!require(grid)) {install.packages("grid"); require(grid)}
if(!require(ggplot2)) {install.packages("ggplot2"); require(ggplot2)}
if(!require(lattice)) {install.packages("lattice"); require(lattice)}
if(!require(gridExtra)) {install.packages("gridExtra"); require(gridExtra)}
if(!require(tidyselect)) {install.packages("tidyselect"); require(tidyselect)}
if(!require(caTools)) {install.packages("caTools"); require(caTools)}
if(!require(nnet)) {install.packages("nnet"); require(nnet)}
if(!require(ROCR)) {install.packages("ROCR"); require(ROCR)}
if(!require(ISLR)) {install.packages("ISLR"); require(ISLR)}
if(!require(MASS)) {install.packages("MASS"); require(MASS)}
if(!require(corrplot)) {install.packages("corrplot"); require(corrplot)}
if(!require(caret)) {install.packages("caret"); require(caret)}
if(!require(class)) {install.packages("class"); require(class)}
if(!require(data.table)) {install.packages("data.table"); require(data.table)}
if(!require(GGally)) {install.packages("GGally"); require(GGally)}
if(!require(car)) {install.packages("car"); require(car)}
if(!require(outliers)) {install.packages("outliers"); require(outliers)}
loan_train <- fread("Train_data.csv")
loan_train = loan_train[,-1]


#Cleaning missing and duplicate data
loan_train <- na.omit(loan_train)
loan_train <- loan_train[!duplicated(loan_train),]
loan_train$Loan_Status <- as.factor(loan_train$Loan_Status)
str(loan_train)

#Checking multicollinearity among variables
loan_train_numeric <- select_if(loan_train, is.numeric)
cor_matrix <- cor(loan_train_numeric) #As none of the correlations is close to 1, we can say that the 
                                      #quantitative variables do not show collinearity.

#Descriptive analysis
summary(loan_train)
sapply(loan_train, sd)

pm <- ggpairs(loan_train, columns = c("Gender", "Married", "Dependents", "Education", "Self_Employed", "Loan_Status"))
pm1 <- ggpairs(loan_train, columns = c("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Loan_Status"))
pm2 <- ggpairs(loan_train, columns = 11:12)
pm
pm1
pm2

#As we can in pm1 there is only one appticant having high income and still no approved loan.
#This is an outlier and we can remove this as an applicant with high income will have less chances of not
#getting his loan approved.

#to remove the outliers we will use the fact that an observation which lies beyond 3 standard deviations
#from the mean is considered to be an outlier.

#outlier for applicant income
app_income_out <- mean(loan_train$ApplicantIncome)+3*sd(loan_train$ApplicantIncome)
app_income_out

#we will remove any observation with applicantincome greater than 24720.22 and not approved loan
loan_train_out <- filter(loan_train, ApplicantIncome > 24720.22 & Loan_Status == "N")
View(loan_train_out)
loan_train <- anti_join(loan_train, loan_train_out)
View(loan_train)

#Splitting training and test data from Given Data
set.seed(123)
index = sample(1:nrow(loan_train), size = .75*nrow(loan_train))
train_data = loan_train[index,]
validation_data = loan_train[-index,]

#Full glm model
glm_fit <- glm(Loan_Status ~ ., data = train_data, family = "binomial")
summary(glm_fit)

glm.probs = predict(glm.fit, validation_data, type = "response")
glm.probs
contrasts(validation_data$Loan_Status)
dim(validation_data)

glm.pred = rep("N", 132)
glm.pred[glm.probs >.7] = "Y"
glm.pred = as.factor(glm.pred)
table(glm.pred, validation_data$Loan_Status)
mean(glm.pred != validation_data$Loan_Status)

pred<-prediction(as.numeric(glm.pred), as.numeric(validation_data$Loan_Status))
perf<-performance(pred,"tpr", "fpr")
plot(perf)
auc<- performance(pred,"auc")
auc

#Model with no predictors
glm_nothing <- glm(Loan_Status ~ 1, data = train_data, family = "binomial")
summary(glm_nothing)

#Backward stepwise model
glm_backward <- step(glm_fit)
summary(glm_backward)

glm.probs_backward = predict(glm_backward, validation_data, type = "response")
glm.pred_backward = rep("N", 132)
glm.pred_backward[glm.probs_backward >.7] = "Y"
glm.pred_backward = as.factor(glm.pred_backward)
table(glm.pred_backward, validation_data$Loan_Status)
mean(glm.pred_backward != validation_data$Loan_Status)
pred_backward<-prediction(as.numeric(glm.pred_backward), as.numeric(validation_data$Loan_Status))
perf_backward<-performance(pred_backward,"tpr", "fpr")
plot(perf_backward)
auc_backward<- performance(pred_backward,"auc")
auc_backward

#Forward stepwise model
glm_forward <-  step(glm_nothing, scope=list(lower=formula(glm_nothing),upper=formula(glm_fit)), direction="forward")
summary(glm_forward)

glm.probs_forward = predict(glm_forward, validation_data, type = "response")
glm.pred_forward = rep("N", 132)
glm.pred_forward[glm.probs_forward >.7] = "Y"
glm.pred_forward = as.factor(glm.pred_forward)
table(glm.pred_forward, validation_data$Loan_Status)
mean(glm.pred_forward != validation_data$Loan_Status)
pred_forward<-prediction(as.numeric(glm.pred_forward), as.numeric(validation_data$Loan_Status))
perf_forward<-performance(pred_forward,"tpr", "fpr")
plot(perf_forward)
auc_forward<- performance(pred_forward,"auc")
auc_forward

#bothways model
glm_bothways = step(glm_nothing, list(lower=formula(glm_nothing),upper=formula(glm_fit)), direction="both", trace = 0)
summary(glm_bothways)

glm.probs_bothways = predict(glm_bothways, validation_data, type = "response")
glm.pred_bothways = rep("N", 132)
glm.pred_bothways[glm.probs_bothways >.7] = "Y"
glm.pred_bothways = as.factor(glm.pred_bothways)
table(glm.pred_bothways, validation_data$Loan_Status)
mean(glm.pred_bothways != validation_data$Loan_Status)
pred_bothways<-prediction(as.numeric(glm.pred_bothways), as.numeric(validation_data$Loan_Status))
perf_bothways<-performance(pred_bothways,"tpr", "fpr")
plot(perf_bothways)
auc_bothways<- performance(pred_bothways,"auc")
auc_bothways

#Model with non-linear transformation
glm_nl_fit <- glm(Loan_Status ~ Credit_History + Property_Area + I(Credit_History^2), data = train_data, family = "binomial")
summary(glm_nl_fit)

glm.probs_nl = predict(glm_nl_fit, validation_data, type = "response")
glm.pred_nl = rep("N", 132)
glm.pred_nl[glm.probs_nl >.7] = "Y"
glm.pred_nl = as.factor(glm.pred_nl)
table(glm.pred_nl, validation_data$Loan_Status)
mean(glm.pred_nl != validation_data$Loan_Status)
pred_nl<-prediction(as.numeric(glm.pred_nl), as.numeric(validation_data$Loan_Status))
perf_nl<-performance(pred_nl,"tpr", "fpr")
plot(perf_nl)
auc_nl<- performance(pred_nl,"auc")
auc_nl

#Model with interaction term
glm_int_fit <- glm(Loan_Status ~ Credit_History + Property_Area + Credit_History:Property_Area, data = train_data, family = "binomial")
summary(glm_int_fit)
glm.probs_int = predict(glm_int_fit, validation_data, type = "response")
glm.pred_int = rep("N", 132)
glm.pred_int[glm.probs_int >.7] = "Y"
glm.pred_int = as.factor(glm.pred_int)
table(glm.pred_int, validation_data$Loan_Status)
mean(glm.pred_int != validation_data$Loan_Status)
pred_int<-prediction(as.numeric(glm.pred_int), as.numeric(validation_data$Loan_Status))
perf_int<-performance(pred_int,"tpr", "fpr")
plot(perf_int)
auc_int<- performance(pred_int,"auc")
auc_int

##knn approach using knn function
#Converting all predictors to numeric
loan_train$Property_Area <- as.numeric(as.factor(loan_train$Property_Area))
train_knn = cbind(loan_train$Credit_History, loan_train$Property_Area)[index,]
test_knn = cbind(loan_train$Credit_History, loan_train$Property_Area)[-index,]
train.Loan_Status <- loan_train$Loan_Status[index]

set.seed(1)
knn.pred = knn(train = train_knn, test = test_knn, cl = train.Loan_Status, k=1)
table(knn.pred, validation_data$Loan_Status)
pred_knn<-prediction(as.numeric(knn.pred), as.numeric(validation_data$Loan_Status))
perf_knn<-performance(pred_knn,"tpr", "fpr")
plot(perf_knn)
auc_knn<- performance(pred_knn,"auc")
auc_knn

##knn approach using caret
# Setting up train controls
repeats = 3
numbers = 10
tunel = 10

set.seed(1234)
x = trainControl(method = 'repeatedcv',
                 number = numbers,
                 repeats = repeats,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

model1 <- train(Loan_Status~Credit_History+Property_Area , data = train_data, method = 'knn',
                preProcess = c('center','scale'),
                trControl = x,
                metric = 'ROC',
                tuneLength = tunel)

# Summary of model
model1
plot(model1)

# Validation
valid_pred <- predict(model1, validation_data, type = 'prob')

#Storing Model Performance Scores
library(ROCR)
pred_val <-prediction(valid_pred[,2],validation_data$Loan_Status)

# Calculating Area under Curve (AUC)
perf_val <- performance(pred_val,'auc')
perf_val

# Plot AUC
perf_val <- performance(pred_val, 'tpr', 'fpr')
plot(perf_val, col = 'green', lwd = 1.5)


##Validation on the given test set using knn model
#KNN model with two predictors (Credit_History and Property_Area) and for k=23 resulted in the 
#best AUC of the ROC Curve, therefore using "model1" to predict values for the given test data
loan_test <- data.table::fread("C:/Users/shrut/OneDrive/Documents/Loan Prediction/Test_data.csv")
loan_test <- as.data.frame(loan_test)
loan_test <- na.omit(loan_test)
loan_test <- loan_test[!duplicated(loan_test),]

valid_pred <- predict(model1,loan_test, type = 'prob')

knn.pred <- rep("N" ,328)
knn.pred[valid_pred[,2] >.7] <- "Y"
loan_test$Loan_Status_Pred <- knn.pred