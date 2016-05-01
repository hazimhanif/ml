require(xgboost)
require(caret)

set.seed(100)

df_train <- read.csv("training_set.csv")
df_test <- read.csv("test_set.csv")

labels <-  df_train["Label"]
df_train = df_train[-grep('Label', colnames(df_train))]


# Reform the data into numerics
labels <- sapply(labels, as.numeric)
df_train <- sapply(df_train, as.numeric)
df_test <- sapply(df_test, as.numeric)

num.class <- 2
labels = as.matrix(as.integer(labels)-1)

param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 20,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)


model <- xgboost(param=param, 
                 data=df_train, 
                 label=labels, 
                 nrounds=25, 
                 verbose=TRUE)

# predict
prediction <- predict(model, df_test)  

# prediction decoding
prediction <- matrix(prediction, nrow=num.class, ncol=length(prediction)/num.class)
prediction <- t(prediction)
prediction <- max.col(prediction, "last")

id <- c(1:1000)
output <- cbind(id,prediction)
colnames(output) <- c("Id","Prediction")
write.csv(output,file="D:/predict_XGBoost.csv",row.names = FALSE)

