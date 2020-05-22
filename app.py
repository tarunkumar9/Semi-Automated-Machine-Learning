"""Flask Login Example and instagram fallowing find"""
from flask import Flask, url_for, render_template, flash, request, redirect, session,logging,request
from flask_sqlalchemy import SQLAlchemy
# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 
from werkzeug.utils import secure_filename
import os
import datetime
import time
# EDA Packages
import pandas as pd 
import numpy as np 
import operator
# ML Packages
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# ML Packages For Vectorization of Text For Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings 
warnings.simplefilter('ignore')


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
Bootstrap(app)


class User(db.Model):
	""" Create user table"""
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(80), unique=True)
	password = db.Column(db.String(80))

	def __init__(self, username, password):
		self.username = username
		self.password = password


@app.route('/', methods=['GET', 'POST'])

def home():
	""" Session control"""
	if not session.get('logged_in'):
		return render_template('index.html')
	else:
		if request.method == 'POST':

			return render_template('index.html') 
		return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
	"""Login Form"""
	if request.method == 'GET':
		return render_template('login.html')
	else:
		name = request.form['username']
		passw = request.form['password']
		try:
			data = User.query.filter_by(username=name, password=passw).first()
			if data is not None:
				session['logged_in'] = True
				return redirect(url_for('home'))
			else:
				return 'Incorrect Login'
		except:
			return "Incorrect Login"

@app.route('/register/', methods=['GET', 'POST'])
def register():
	"""Register Form"""
	if request.method == 'POST':
		new_user = User(username=request.form['username'], password=request.form['password'])
		db.session.add(new_user)
		db.session.commit()
		return render_template('login.html')
	return render_template('register.html')

@app.route("/logout")
def logout():
	"""Logout Form"""
	session['logged_in'] = False
	session.pop('username',None)  
	return redirect(url_for('home'))
	return render_template('logout.html')

# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		model_type = request.form['model_type']
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
		# os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB',filename))
		fullfile = os.path.join('static/uploadsDB',filename)
		# For Time
		date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))
		# EDA function
		df = pd.read_csv(os.path.join('static/uploadsDB',filename))
		df_size = df.size
		df_shape = df.shape
		df_dtypes=df.dtypes
		df_describe=df.describe()
		# Unique values in each variable of train dataset
		df_nunique=df.nunique()
		df_missing=df.isnull().sum().sort_values(ascending=False)
		df_columns = list(df.columns)
		df.fillna(-999, inplace=True)
		df=df.apply(le.fit_transform)
		df_targetname = df[df.columns[-1]].name
		df_featurenames = df_columns[0:-1] # select all columns till last column
		df_Xfeatures = df.iloc[:,0:-1] 
		df_Ylabels = df[df.columns[-1]] # Select the last column as target
		# same as above df_Ylabels = df.iloc[:,-1]
		# Model Building
		X = df_Xfeatures
		Y = df_Ylabels
		seed = 7
		models = []
		# prepare models
		if model_type=='classification':
			models.append(('LogisticRegression', LogisticRegression()))
			models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
			models.append(('KNeighborsClassifier', KNeighborsClassifier()))
			models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
			models.append(('GaussianNB', GaussianNB()))
			models.append(('MLPClassifier', MLPClassifier()))
			models.append(('RandomForestClassifier', RandomForestClassifier()))
			results = []
			names = []
			allmodels = []
			compare=[]
			res = {} 
			prediction=[]
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
				results.append(cv_results)
				names.append(name)
				msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
				allmodels.append(msg)
				compare.append(cv_results.mean())
				# Making predictions for the test data
				model.fit(X, Y)
				pred=model.predict_proba(X)[:,1]
				model_results = results
				model_names = names
				prediction.append(pred)
			for key in model_names: 
				for value in compare: 
					res[key] = value 
					compare.remove(value) 
					break
			res=dict(sorted(res.items(), key=operator.itemgetter(1),reverse=True))  
		elif model_type=='regression':
			df_size = df.size
			df_shape = df.shape
			df_dtypes=df.dtypes
			df_describe=df.describe()
			df_columns = list(df.columns)
			df.fillna(-999, inplace=True)
			df=df.apply(le.fit_transform)
			df_targetname = df[df.columns[-1]].name
			df_featurenames = df_columns[0:-1] # select all columns till last column
			df_Xfeatures = df.iloc[:,0:-1] 
			df_Ylabels = df[df.columns[-1]] # Select the last column as target
			# same as above df_Ylabels = df.iloc[:,-1]
			# Model Building
			X = df_Xfeatures
			Y = df_Ylabels
			seed = 7
			models = []
			models.append(('LogisticRegression', LogisticRegression()))
			models.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
			models.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
			models.append(('MLPRegressor', MLPRegressor()))
			models.append(('RandomForestRegressor', RandomForestRegressor()))
			# evaluate each model in turn
			results = []
			names = []
			allmodels = []
			prediction=[]
			compare=[]
			scoring = 'accuracy'
			for name, model in models:
				kfold = model_selection.KFold(n_splits=10, random_state=seed)
				model.fit(X, Y)
				cv_results=model.score(X,Y)
				results.append(cv_results)
				names.append(name)
				msg = "%s: %f (%f)" % (name, cv_results, cv_results.std())
				allmodels.append(msg)
				model_results = allmodels
				model_names = names
				pred=model.predict(X)
				prediction.append(pred)
				compare.append(cv_results)
				model_results = results
				model_names = names
				print(cv_results)
			res = {} 
			for key in model_names: 
				for value in compare: 
					res[key] = value 
					compare.remove(value) 
					break
			res=dict(sorted(res.items(), key=operator.itemgetter(1),reverse=True)) 	
					
	return render_template('details.html',filename=filename,date=date,
		df_size=df_size,
		df_shape=df_shape,
		df_dtypes=df_dtypes,
		df_columns =df_columns,
		df_targetname =df_targetname,
		model_results = allmodels,
		model_names = names,
		fullfile = fullfile,
		dfplot = df,
		models=models,
		res=res,
		df_describe=df_describe,
		df_nunique=df_nunique,
		df_missing=df_missing,
		prediction=prediction
		)
if __name__ == '__main__':
	app.debug = True
	db.create_all()
	app.secret_key = "123"
	app.run(host='127.0.0.1')
	
