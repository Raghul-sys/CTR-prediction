from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('newmodel.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    x = request.form['states']
    state=np.array(['State_Alabama','State_Alaska','State_Arizona','State_Arkansas','State_California','State_Colorado','State_Connecticut','State_Delaware','State_District of Columbia','State_Florida','State_Georgia','State_Hawaii','State_Idaho','State_Illinois','State_Indiana','State_Iowa','State_Kansas','State_Kentucky','State_Louisiana','State_Maine','State_Maryland','State_Massachusetts','State_Michigan','State_Minnesota','State_Mississippi','State_Missouri','State_Montana','State_Nebraska','State_Nevada','State_New Hampshire','State_New Jersey','State_New Mexico','State_New York','State_North Carolina','State_North Dakota','State_Ohio','State_Oklahoma','State_Oregon','State_Pennsylvania','State_Rhode Island','State_South Carolina','State_South Dakota','State_Tennessee','State_Texas','State_Utah','State_Vermont','State_Virginia','State_Washington','State_West Virginia','State_Wisconsin','State_Wyoming'])
    def states(x):
        if x in state:
            return np.where(state==x, 1, 0)
    statenew = states(x)

    y = request.form['Exchange']
    exchange=np.array(['exc_Aol','exc_AppNexus','exc_FreeWheel','exc_GoogleAdx','exc_Index','exc_Lkqd','exc_MoPub','exc_Nexage','exc_OpenX','exc_Revvid','exc_Rubicon','exc_SpotX2','exc_Telaria'])
    def exchanges(y):
        if y in exchange:
            return np.where(exchange==y, 1, 0)  
    excnew = exchanges(y)

    z =request.form['adtype']
    adtype=np.array(['adtype_1','adtype_2','adtype_3'])
    def adtypes(z):
        if z in adtype:
            return np.where(adtype==z,1,0)
    adtypenew = adtypes(z)

    b=np.append(statenew,excnew)
    c=np.append(b,adtypenew)   





    prediction = model.predict([c])


    return render_template('index.html',prediction_text="The predicted clicks is {}".format(prediction))


    


if __name__ == "__main__":
    app.run(debug=True)
