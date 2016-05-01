require(xgboost)
require(caret)

set.seed(100)

df_train <- read.csv("Training_NoID(Complete).csv",stringsAsFactors = FALSE)
df_test <- read.csv("Testing_NoID(Complete).csv",stringsAsFactors = FALSE)

# change output colum(Class) to number
df_train$Class <- replace(df_train$Class,df_train$Class=="legitimate",2)
df_train$Class <- replace(df_train$Class,df_train$Class=="spam",1)
df_test$Class <- replace(df_test$Class,df_test$Class=="legitimate",2)
df_test$Class <- replace(df_test$Class,df_test$Class=="spam",1)


# Remove isolate and remove labels from training set
train_labels <-  df_train["Class"]
test_labels <-  df_test["Class"]
df_train = df_train[-grep('Class', colnames(df_train))]
df_test = df_test[-grep('Class', colnames(df_test))]



# Reform the data into numerics
train_labels <- sapply(train_labels, as.numeric)
test_labels <- sapply(test_labels, as.numeric)
df_train <- sapply(df_train, as.numeric)
df_test <- sapply(df_test, as.numeric)

# Number of class
num.class <- 2
train_labels = as.matrix(as.integer(train_labels)-1)

param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 6,    # maximum depth of tree 
              "eta" = 0.001,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 1  # minimum sum of instance weight needed in a child 
)


model <- xgboost(param=param, 
                 data=df_train, 
                 label=train_labels, 
                 nrounds=25, 
                 verbose=TRUE)

# predict
prediction <- predict(model, df_test)  


# prediction decoding
prediction <- matrix(prediction, nrow=num.class, ncol=length(prediction)/num.class)
prediction <- t(prediction)
prediction <- max.col(prediction, "last")

# confusion matrix
confusionMatrix(factor(test_labels), factor(prediction))
