## Title: Prediction of Titanic Survival

# Load packages
library('ggplot2')
library('ggthemes') 
library('scales')
library('rpart')  
library('randomForest')
library('caret')       
library('doParallel')   
library('nnet')      
library('kernlab')  
library('e1071')  
library('Hmisc')     
library('stringr')    
library('corrgram')   
library('ROCR')   
library('dplyr')



#-------------------------------------------------------------------------------------------------------
## Load in Data

train <- as_data_frame(read.csv('train.csv', stringsAsFactors = F))
test  <- as_data_frame(read.csv('test.csv', stringsAsFactors = F))


#-------------------------------------------------------------------------------------------------------
## Combine Test and Train Dataset
titanic <- bind_rows(train, test) # bind training & test data


#-------------------------------------------------------------------------------------------------------

titanic$Title <- NA
titanic$Title <- str_sub(titanic$Name, str_locate(titanic$Name, ",")[, 1] + 2, str_locate(titanic$Name, "\\.")[,1]-1)

## Title Before
table(titanic$Title)

## Make Miss / Mrs Consistent
titanic$Title[titanic$Title == 'Mlle'] <- 'Miss' 
titanic$Title[titanic$Title == 'Ms']   <- 'Miss'
titanic$Title[titanic$Title == 'Mme']  <- 'Mrs'

## Place Common Titles in a Vector
common_titles = c('Mr', 'Miss', 'Mrs', 'Master')

## Rename title to 'Uncommon' if it is not in common_titles
titanic$Title[!titanic$Title %in% common_titles]  <- 'Uncommon'

titles_list = table(titanic$Sex, titanic$Title)

## Title After
table(titanic$Title)

#-------------------------------------------------------------------------------------------------------
## Impute missing values on Embarked with the mode, 'S'

titanic$Embarked[is.na(titanic$Embarked)] <- 'S'

## For some reason this did not work as well as i'd hope so i had to change them manually

titanic$Embarked[830] <- 'S'
titanic$Embarked[62] <- 'S'

#-------------------------------------------------------------------------------------------------------
## Impute missing value for Fare with the median for that Pclass
titanic$Fare[1044] <- median(titanic$Fare[titanic$Pclass == '3' & titanic$Embarked == 'S'], na.rm = TRUE)


#-------------------------------------------------------------------------------------------------------


## Age Imputation
fit.Age<-rpart(Age~Pclass+Title+Sex+FamilySize+Fare+Embarked+FamilySizeCat,data=subset(titanic,Age!=is.na(Age)))
rpartAge<- titanic$Age
rpartAge[is.na(titanic$Age)]<-predict(fit.Age,titanic[is.na(titanic$Age),])
titanic$Age <- rpartAge

## Age Summary After
summary(titanic$Age)


#-------------------------------------------------------------------------------------------------------
## Create New Columns: FamilySize & FamilySizeCat

titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$FamilySizeCat <- NA
titanic$FamilySizeCat <- factor(titanic$FamilySizeCat)
titanic$FamilySizeCat <- ifelse(titanic$FamilySize == 1, "Small", 
ifelse(titanic$FamilySize <= 4, "Medium", "Large")) 
titanic$FamilySizeCat <- factor(titanic$FamilySizeCat)


#-------------------------------------------------------------------------------------------------------
## Create New Column: AgeCat

titanic$AgeCat <- 'NA'
titanic$AgeCat[titanic$Age < 18] <- 'Child'
titanic$AgeCat[titanic$Age >= 18] <- 'Adult'



#-------------------------------------------------------------------------------------------------------
## Turn Non-Numerical values into Factors

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','FamilySizeCat')

titanic[factor_vars] <- lapply(titanic[factor_vars], function(x) as.factor(x))


#-------------------------------------------------------------------------------------------------------
## Split the titanic dataset back into Train and Test

train <- titanic[1:891,]
test <- titanic[892:1309,]

dim(train); dim(test)


#-------------------------------------------------------------------------------------------------------

## Visualize the relationship between Pclass & survival

ggplot(data=train, aes(factor(Survived), fill=factor(Pclass))) + labs(fill='PCLASS') + xlab('SURVIVED') + 
ylab('COUNT OF PASSENGER') + ggtitle("SURVIVAL BY PASSENGER CLASS") + geom_bar(position="dodge") 


## Visualize the relationship between Title & survival

ggplot(data=train, aes(factor(Survived), fill=factor(Title))) + labs(fill='TITLE') + xlab('SURVIVED') + 
ylab('COUNT OF PASSENGER') + ggtitle("SURVIVAL BY TITLE") + geom_bar(position="dodge") 


## Visualize the relationship between Gender & survival

ggplot(data=train, aes(factor(Survived), fill=factor(Sex))) + labs(fill='GENDER') + xlab('SURVIVED') + 
ylab('COUNT OF PASSENGER') + ggtitle("SURVIVAL BY GENDER") + geom_bar(position="dodge") 


## Visualize the relationship between FamilySizeCat & survival

ggplot(data=train, aes(factor(Survived), fill=factor(FamilySizeCat))) + labs(fill='FAMILYSIZECAT') + xlab('SURVIVED') + 
ylab('COUNT OF PASSENGER')+ ggtitle("SURIVAL BY FAMILYSIZECAT")+ geom_bar(position="dodge") 


## Visualize the relationship between Embarked & survival

ggplot(data=train, aes(factor(Survived), fill=factor(Embarked))) + labs(fill='EMBARKED') + xlab('SURVIVED') + 
ylab('COUNT OF PASSENGER') + ggtitle("SURVIVAL BY EMBARKED") + geom_bar(position="dodge") 


## Visualize the relationship between Fare & Survival

ggplot(data=train, aes(Fare, Survived)) + ggtitle("SURVIVAL BY FARE ($)") + xlab('FARE') + 
ylab('SURVIVED')+ geom_jitter(height=0.08, alpha=0.4) + stat_smooth(method="loess", alpha=0.2, col="red") 


## Visualize the relationship between Age & survival

ggplot(data=train, aes(Age, Survived)) + ggtitle("SURVIVAL BY AGE") + xlab('AGE') + 
ylab('SURVIVED') + geom_jitter(height=0.08, alpha=0.4) + stat_smooth(method="loess", alpha=0.2, col="red") 



#-------------------------------------------------------------------------------------------------------
## Model Building uisng randomForest

set.seed(754)

RFmodel <- randomForest(factor(Survived) ~ Pclass + Sex + Age + 
Fare + Embarked + Title + FamilySizeCat, data = train)

#-------------------------------------------------------------------------------------------------------
## Get importance

importance    <- importance(RFmodel)
varImportance <- data.frame(Variables = row.names(importance), 
Importance = round(importance[ ,'MeanDecreaseGini'],2))

## Create Rank

rankImportance <- varImportance %>%
mutate(Rank = paste0('#',dense_rank(desc(Importance))))

## Visualizing the Importance of Variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance),y = Importance, fill = Importance))+ 
ggtitle("IMPORTANCE OF VARIABLES RELATING TO SURVIVAL") + geom_bar(stat='identity') + 
geom_text(aes(x = Variables, y = 1, label = Rank), hjust=0, vjust=2, size = 5, colour = 'White') + 
xlab('VARIABLES') + ylab('IMPORTANCE') + labs(fill='IMPORTANCE') + coord_flip() + theme_few()


#-------------------------------------------------------------------------------------------------------

## Saving RFmodel Results to a CSV File
prediction <- predict(RFmodel, test)
submission <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
write.csv(submission,'submission.csv',row.names=F)
