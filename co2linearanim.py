from collections import deque
from matplotlib.animation import FuncAnimation
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df=pd.read_csv("FuelConsumptionCo2.csv")
selected_df = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

mask=np.random.rand(len(df))<0.8
train=selected_df[mask]
test=selected_df[~mask]

def engsize():
    regression=linear_model.LinearRegression()
    train_x=np.asanyarray(train[["ENGINESIZE"]])
    train_y=np.asanyarray((train[["CO2EMISSIONS"]]))
    regression.fit(train_x,train_y)

    print ('Coefficients: ', regression.coef_[0][0])
    print ('Intercept: ',regression.intercept_[0])

    fig=plt.figure()
    fig.suptitle("TRAIN DATA")

    ax = plt.axes(xlim=(0,10),ylim=(0,500))
    colors=deque(['blue','yellow','green','cyan','black'])

    def update(i):
        if i in train.ENGINESIZE.keys():
            ax.scatter(train.ENGINESIZE[i], train.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>130:
            ax.scatter(train.ENGINESIZE[0:200], train.CO2EMISSIONS[0:200],  color='blue')
            ax.scatter(train.ENGINESIZE[200:400], train.CO2EMISSIONS[200:400],  color='yellow')
            ax.scatter(train.ENGINESIZE[400:600], train.CO2EMISSIONS[400:600],  color='green')
            ax.scatter(train.ENGINESIZE[600:800], train.CO2EMISSIONS[600:800],  color='cyan')
            ax.scatter(train.ENGINESIZE[800:], train.CO2EMISSIONS[800:],  color='black')
            anim.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    prediction=regression.predict(test_x)

    fig1=plt.figure()
    fig1.suptitle("TEST DATA")

    ax1 = plt.axes(xlim=(0,10),ylim=(0,500))

    def updatetest(i):
        if i in test.ENGINESIZE.keys():
            ax1.scatter(test.ENGINESIZE[i], test.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax1.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>150:
            ax1.scatter(test.ENGINESIZE[0:50], test.CO2EMISSIONS[0:50],  color='blue')
            ax1.scatter(test.ENGINESIZE[50:100], test.CO2EMISSIONS[50:100],  color='yellow')
            ax1.scatter(test.ENGINESIZE[100:150], test.CO2EMISSIONS[100:150],  color='green')
            ax1.scatter(test.ENGINESIZE[150:], test.CO2EMISSIONS[150:],  color='black')
            anim1.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)
    anim1=FuncAnimation(fig1,updatetest,interval=1)

    test_y=np.squeeze(test_y)
    prediction=np.squeeze(prediction)

    pred_df=pd.DataFrame({"ACTUAL":test_y ,"PREDICTED":prediction,"DIFFERENCE":prediction-test_y})
    print(pred_df.to_string())

    print(f"RMSE: {np.mean((prediction-test_y)**2)**(1/2):.2f}")
    print(f"R2 %: {r2_score(test_y,prediction)*100:.2f}")

    plt.show()
    custom_x=int(input("Predict CO2 Emission for custom value of Engine Size:"))
    print(f"CO2 Emission for {custom_x} Engine Size is {regression.coef_[0][0]*custom_x+regression.intercept_[0]}")

def cylinders():
    regression=linear_model.LinearRegression()
    train_x=np.asanyarray(train[["CYLINDERS"]])
    train_y=np.asanyarray((train[["CO2EMISSIONS"]]))
    regression.fit(train_x,train_y)

    print ('Coefficients: ', regression.coef_[0][0])
    print ('Intercept: ',regression.intercept_[0])

    fig=plt.figure()
    fig.suptitle("TRAIN DATA")

    ax = plt.axes(xlim=(0,14),ylim=(0,500))
    colors=deque(['blue','yellow','green','cyan','black'])

    def update(i):
        if i in train.CYLINDERS.keys():
            ax.scatter(train.CYLINDERS[i], train.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>130:
            ax.scatter(train.CYLINDERS[0:200], train.CO2EMISSIONS[0:200],  color='blue')
            ax.scatter(train.CYLINDERS[200:400], train.CO2EMISSIONS[200:400],  color='yellow')
            ax.scatter(train.CYLINDERS[400:600], train.CO2EMISSIONS[400:600],  color='green')
            ax.scatter(train.CYLINDERS[600:800], train.CO2EMISSIONS[600:800],  color='cyan')
            ax.scatter(train.CYLINDERS[800:], train.CO2EMISSIONS[800:],  color='black')
            anim.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)

    test_x = np.asanyarray(test[['CYLINDERS']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    prediction=regression.predict(test_x)

    fig1=plt.figure()
    fig1.suptitle("TEST DATA")

    ax1 = plt.axes(xlim=(0,14),ylim=(0,500))

    def updatetest(i):
        if i in test.CYLINDERS.keys():
            ax1.scatter(test.CYLINDERS[i], test.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax1.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>150:
            ax1.scatter(test.CYLINDERS[0:50], test.CO2EMISSIONS[0:50],  color='blue')
            ax1.scatter(test.CYLINDERS[50:100], test.CO2EMISSIONS[50:100],  color='yellow')
            ax1.scatter(test.CYLINDERS[100:150], test.CO2EMISSIONS[100:150],  color='green')
            ax1.scatter(test.CYLINDERS[150:], test.CO2EMISSIONS[150:],  color='black')
            anim1.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)
    anim1=FuncAnimation(fig1,updatetest,interval=1)

    test_y=np.squeeze(test_y)
    prediction=np.squeeze(prediction)

    pred_df=pd.DataFrame({"ACTUAL":test_y ,"PREDICTED":prediction,"DIFFERENCE":prediction-test_y})
    print(pred_df.to_string())

    print(f"RMSE: {np.mean((prediction-test_y)**2)**(1/2):.2f}")
    print(f"R2 %: {r2_score(test_y,prediction)*100:.2f}")

    plt.show()
    custom_x=int(input("Predict CO2 Emission for custom value of Number of Cylinders:"))
    print(f"CO2 Emission for {custom_x} Number of Cylinders is {regression.coef_[0][0]*custom_x+regression.intercept_[0]}")

def fuelconsp():
    regression=linear_model.LinearRegression()
    train_x=np.asanyarray(train[["FUELCONSUMPTION_COMB"]])
    train_y=np.asanyarray((train[["CO2EMISSIONS"]]))
    regression.fit(train_x,train_y)

    print ('Coefficients: ', regression.coef_[0][0])
    print ('Intercept: ',regression.intercept_[0])

    fig=plt.figure()
    fig.suptitle("TRAIN DATA")

    ax = plt.axes(xlim=(0,30),ylim=(0,500))
    colors=deque(['blue','yellow','green','cyan','black'])

    def update(i):
        if i in train.FUELCONSUMPTION_COMB.keys():
            ax.scatter(train.FUELCONSUMPTION_COMB[i], train.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>130:
            ax.scatter(train.FUELCONSUMPTION_COMB[0:200], train.CO2EMISSIONS[0:200],  color='blue')
            ax.scatter(train.FUELCONSUMPTION_COMB[200:400], train.CO2EMISSIONS[200:400],  color='yellow')
            ax.scatter(train.FUELCONSUMPTION_COMB[400:600], train.CO2EMISSIONS[400:600],  color='green')
            ax.scatter(train.FUELCONSUMPTION_COMB[600:800], train.CO2EMISSIONS[600:800],  color='cyan')
            ax.scatter(train.FUELCONSUMPTION_COMB[800:], train.CO2EMISSIONS[800:],  color='black')
            anim.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)

    test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    prediction=regression.predict(test_x)

    fig1=plt.figure()
    fig1.suptitle("TEST DATA")

    ax1 = plt.axes(xlim=(0,30),ylim=(0,500))

    def updatetest(i):
        if i in test.FUELCONSUMPTION_COMB.keys():
            ax1.scatter(test.FUELCONSUMPTION_COMB[i], test.CO2EMISSIONS[i],  color=colors[0])
            colors.rotate(np.random.randint(1,len(colors)))
        else:
            pass
        if i>100:
            ax1.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
        if i>150:
            ax1.scatter(test.FUELCONSUMPTION_COMB[0:50], test.CO2EMISSIONS[0:50],  color='blue')
            ax1.scatter(test.FUELCONSUMPTION_COMB[50:100], test.CO2EMISSIONS[50:100],  color='yellow')
            ax1.scatter(test.FUELCONSUMPTION_COMB[100:150], test.CO2EMISSIONS[100:150],  color='green')
            ax1.scatter(test.FUELCONSUMPTION_COMB[150:], test.CO2EMISSIONS[150:],  color='black')
            anim1.event_source.stop()

    anim = FuncAnimation(fig, update, interval=1)
    anim1=FuncAnimation(fig1,updatetest,interval=1)

    test_y=np.squeeze(test_y)
    prediction=np.squeeze(prediction)

    pred_df=pd.DataFrame({"ACTUAL":test_y ,"PREDICTED":prediction,"DIFFERENCE":prediction-test_y})
    print(pred_df.to_string())

    print(f"RMSE: {np.mean((prediction-test_y)**2)**(1/2):.2f}")
    print(f"R2 %: {r2_score(test_y,prediction)*100:.2f}")

    plt.show()
    custom_x=int(input("Predict CO2 Emission for custom value of Total Fuel Consumption:"))
    print(f"CO2 Emission for {custom_x} Total Fuel Consumption is {regression.coef_[0][0]*custom_x+regression.intercept_[0]}")

selected_constraint=int(input("""
Please select constraint for predicting CO2 emissions.:
1. Engine Size (Press 1)
2. No. of Cylinders (Press 2)
3. Total fuel Conumption (Press 3)
"""))

if selected_constraint==1:
    engsize()
elif selected_constraint==2:
    cylinders()
else:
    fuelconsp()