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
loan_train <- data.table::fread("C:/Users/shrut/OneDrive/Documents/Loan Prediction/Train_data.csv")
loan_train <- as.data.frame(loan_train)



#Cleaning missing and duplicate values
loan_train <- na.omit(loan_train)
loan_train <- loan_train[!duplicated(loan_train),]
loan_train$Loan_Status <- as.factor(loan_train$Loan_Status)

#Splitting training and test data from Given Data
set.seed(125)
split <- sample.split(loan_train$Loan_Status, SplitRatio = .75)
train_data <- subset(loan_train, split == TRUE)
test_data <- subset(loan_train, split == FALSE)


#Visualizing data
str(train_data)
num_cols <- c("Dependents", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", 
              "Credit_History")
plot_box <- function(df, cols, col_x = "Loan_Status"){
  options(repr.plot.width = 4, repr.plot.height = 3.5) #Set the initial plot area dimensions
  for(col in cols){
    p = ggplot(df, aes_string(col_x, col)) +
      geom_boxplot() + 
      ggtitle(paste('Box plot of', col, '\n vs.', col_x))
    print(p)
  }
  
}

plot_box(train_data, num_cols)

plot_violin <- function(df, cols, col_x = "Loan_Status"){
  options(repr.plot.width = 4, repr.plot.height = 3.5) #Set the initial plot area dimensions
  for(col in cols){
    p = ggplot(df, aes_string(col_x, col)) +
      geom_violin() + 
      ggtitle(paste('Violin plot of', col, '\n vs.', col_x))
    print(p)
  }
  
}

plot_violin(train_data, num_cols)

cat_cols <- c("Gender", "Married", "Education", "Self_Employed", "Property_Area")
plot_bars <- function(df, cat_cols){
  options(repr.plot.width = 6, repr.plot.height = 5)
  temp1 = df[df$Loan_Status == 'N',]
  temp2 = df[df$Loan_Status == 'Y',]
  for(col in cat_cols){
    p1 = ggplot(temp1, aes_string(col))+
      geom_bar()+
      ggtitle(paste('Bar plot of \n', col, 'for not approved loans'))+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
    p2 = ggplot(temp2, aes_string(col))+
      geom_bar()+
      ggtitle(paste('Bar plot of \n', col, 'for approved loans'))+
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
    grid.arrange(p1, p2, nrow = 1)
  }
}

plot_bars(train_data, cat_cols)

##fitting a logistic regression model without some variables on the basis of Visualizations
glm.fit <- glm(Loan_Status ~ ApplicantIncome + Loan_Amount_Term + Credit_History + Married + Property_Area + 
                 CoapplicantIncome*LoanAmount, data = train_data, 
               family = "binomial")
summary(glm.fit)
glm.probs <- predict(glm.fit, test_data, type = "response")
glm.pred <- rep("N" ,133)
glm.pred[glm.probs >.7] <- "Y"
test_data$Loan_Status_Pred <- glm.pred
table(glm.pred, test_data$Loan_Status)

#Measure the model fitness
test_data$Loan_Status_Pred <- as.factor(test_data$Loan_Status_Pred)
str(test_data)
pred<-prediction(as.numeric(test_data$Loan_Status_Pred), as.numeric(test_data$Loan_Status))
perf<-performance(pred,"tpr", "fpr")
plot(perf)
auc<- performance(pred,"auc")
auc

##fitting a logistic regression model with all the variables
glm.fit1 <- glm(Loan_Status ~ Dependents + ApplicantIncome + CoapplicantIncome + LoanAmount + Loan_Amount_Term + Credit_History
               + Married + Education + Property_Area + Gender + Self_Employed + CoapplicantIncome*LoanAmount +
                 ApplicantIncome*LoanAmount, 
               data = train_data, family = "binomial")
summary(glm.fit1)
glm.probs1 <- predict(glm.fit1, test_data, type = "response")
glm.pred1 <- rep("N" ,133)
glm.pred1[glm.probs1 >.6] <- "Y"
test_data$Loan_Status_Pred1 <- glm.pred1
table(glm.pred1, test_data$Loan_Status)

#Measure the model fitness
test_data$Loan_Status_Pred1 <- as.factor(test_data$Loan_Status_Pred1)
str(test_data)
pred1<-prediction(as.numeric(test_data$Loan_Status_Pred1), as.numeric(test_data$Loan_Status))
perf1<-performance(pred1,"tpr", "fpr")
plot(perf1)
auc<- performance(pred1,"auc")
auc

##Knn approach
loan_train <- loan_train[-1]
library (class)
set.seed(101)

#Partitioning the data into training and validation data
index = createDataPartition(loan_train$Loan_Status, p = 0.7, list = F )
train_knn = loan_train[index,]
validation_knn = loan_train[-index,]

# Setting levels for both training and validation data
levels(train_knn$Loan_Status) <- make.names(levels(factor(train_knn$Loan_Status)))
levels(validation_knn$Loan_Status) <- make.names(levels(factor(validation_knn$Loan_Status)))

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

model1 <- train(Loan_Status~. , data = train_knn, method = 'knn',
                preProcess = c('center','scale'),
                trControl = x,
                metric = 'ROC',
                tuneLength = tunel)

# Summary of model
model1
plot(model1)

# Validation
valid_pred <- predict(model1,validation_knn, type = 'prob')

#Storing Model Performance Scores
library(ROCR)
pred_val <-prediction(valid_pred[,2],validation_knn$Loan_Status)

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

