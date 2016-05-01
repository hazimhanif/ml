require(xgboost)
require(caret)

set.seed(100)

df_train <- read.csv("Training_NoID(Complete).csv")

train_labels <-  df_train["Class"]
df_train = df_train[-grep('Class', colnames(df_train))]


# Reform the data into numerics
train_labels <- sapply(train_labels, as.numeric)
df_train <- sapply(df_train, as.numeric)

num.class <- 2
train_labels = as.matrix(as.integer(train_labels)-1)

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


model <- xgb.cv(params=param,
              data=data.matrix(df_train), 
              label=train_labels,
              nfold=2, 
              nrounds=50,
              prediction=TRUE, 
              verbose=TRUE)


# get CV's prediction decoding
prediction = matrix(model$pred, 
                 nrow=length(model$pred)/num.class, 
                 ncol=num.class)

prediction = max.col(prediction, "last")



# confusion matrix
confusionMatrix(factor(train_labels+1), factor(prediction))

