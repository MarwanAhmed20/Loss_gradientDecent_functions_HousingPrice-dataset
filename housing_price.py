import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loss function (mean square error function).
def loss(x_train, y_train, c1, c2, c3, c4):
    n = len(x_train)  # get the length of the training set.
    y_predicted = c2 * x_train.iloc[:, 0] + c3 * x_train.iloc[:, 1] + c4 * x_train.iloc[:, 2] + c1  # calculate the prediction.
    cost = (1 / n) * sum([val**2 for val in (y_train - y_predicted)])  # calculate the mean square error cost.


# gradient decent function.
def gradientDecent(x_train, y_train):
    learn_rate = 0.001
    n = len(x_train)
    c1 = 6  
    c2 = 5 
    c3 = 3  
    c4 = 1.5  

    
    y_predicted = c2 * x_train.iloc[:, 0] + c3 * x_train.iloc[:, 1] + c4 * x_train.iloc[:, 2] + c1

    # minimize the error with gradient decent
    for i in range(150):
        c1 = c1 - learn_rate * (-(2 / n) * sum(y_train - y_predicted))
        c2 = c2 - learn_rate * (-(2 / n) * sum(x_train.iloc[:, 0] * (y_train - y_predicted)))
        c3 = c3 - learn_rate * (-(2 / n) * sum(x_train.iloc[:, 1] * (y_train - y_predicted)))
        c4 = c4 - learn_rate * (-(2 / n) * sum(x_train.iloc[:, 2] * (y_train - y_predicted)))
        loss(x_train, y_train, c1, c2, c3, c4)
    return c1, c2, c3, c4


#  print the test wights
def printWight(c1, c2, c3, c4):
    y_predicted = c2 * x_test.iloc[:, 0] + c3 * x_test.iloc[:, 1] + c4 * x_test.iloc[:, 2] + c1
    return y_predicted

# function to calculate accuracy
def accuracy(test, predict):
    right = 0  # record to the right prediction
    for i in range(len(test)):  # loop throw the test label
        if test.index[i] == predict.index[i]:  # if it is the right answer
            right += 1  # increment by one
    print("Accuracy of the model is : ", right / float(len(test)) * 100)

# load data
dataset = pd.read_csv(r"D:\marwan\FCI\ML\ASS\USA_Housing.csv")
dataset.head()

standard_x = StandardScaler()
x_train = standard_x.fit_transform(dataset)
x_test = standard_x.fit_transform(dataset) 

x_data = dataset[['AreaIncome', 'AreaNumberofRooms', 'AreaPopulation']]
y_data = dataset['Price']

# test train split built in function.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
c1, c2, c3, c4 = gradientDecent(x_train, y_train)




print(printWight(c1, c2, c3, c4))
accuracy(y_test, printWight(c1, c2, c3, c4))

plt.scatter(y_test, printWight(c1, c2, c3, c4))
plt.show()
