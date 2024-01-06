from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
app = Flask(__name__)

@app.route('/home')
def home():
    model= LogisticRegression()
    X_train,X_test,y_train,y_test = train_test_split(df.data,df.target)
    model.fit(x_train,y_train)
    pickle.dumps(model,open("breast_cancer.pkl","wb"))
    return render_template("result.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    data = request.form['data']
    form_array = np.array([[data]])
    model= pickle.load(open("breast_cancer.pkl","rb"))
    prediction =model.predict(form_array)[0]
    if prediction==0:
        result= "THE BREAST CANCER IS MALIGNANT"
    elif prediction==1:
        result= "THE BREAST CANCER IS BENIGN"
    else:
        result="NO CANCER"
    return render_template("result.html",result= result)

if __name__== '__main__':
    app.run(debug=True)