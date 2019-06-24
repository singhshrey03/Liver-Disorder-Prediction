from flask import Flask, session, redirect, url_for, escape, request, render_template, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/liver', methods=['GET','POST'])
def Liver():
    if request.method == 'POST':
        mcv=request.form['mcv']
        alkphos=request.form['alkphos']
        sgpt=request.form['sgpt']
        sgot=request.form['sgot']
        gammagt=request.form['gammagt']
        drinks=request.form['drinks']
        temp =TEST(mcv, alkphos, sgpt, sgot, gammagt, drinks)
        return render_template('output.html',temp = temp)
    return render_template('input.html')

def data_preprocessing():
	dataset = pd.read_csv('liver disorder.csv')
	return (dataset)

def TEST(mcv, alkphos, sgpt, sgot, gammagt, drinks):
	dataset = data_preprocessing()
	X = dataset.iloc[:,0:6].values
	y = dataset.iloc[:,6].values
	
	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X = sc_X.fit_transform(X)
	
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators = 625, criterion = 'entropy', random_state = 0)
	classifier.fit(X, y)

	test1 = int(mcv)
	test2 = int(alkphos)
	test3 = int(sgpt)
	test4 = int(sgot)
	test5 = int(gammagt)
	test6 = float(drinks)

	result = classifier.predict(sc_X.transform(np.array([[test1,test2,test3,test4,test5,test6]])))

	if result == 0:
		temp = "You are not suffering from any Liver disease or disorder."
	else :
		temp = "You might be suffering from Liver disease or disorder because of your excessive drinking."

	return temp

if __name__ == '__main__':
    app.run()

