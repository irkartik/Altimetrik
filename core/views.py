from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response

from .serializers import AccuracySerializer

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class calculate(views.APIView):

	def post(self, request):
		csv_file = request.FILES['file']
		
		data = pd.read_csv(csv_file)

		data["Product_Info_2"] = data["Product_Info_2"].replace({"A1": 11, "A2": 12, "A3": 13, "A4": 14, "A5": 15, "A6": 16, "A7": 17, "A8": 18, "B1": 21, "B2": 22, "C1": 31, "C2":32, "C3": 33, "C4": 34, "D1": 41, "D2": 42, "D3": 43, "D4": 44, "E1":51})


		X = data.iloc[:, 1:127].values
		y = data.iloc[:, 127:128].values


		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

		xgb1 = xgb.XGBClassifier(learning_rate =0.01, n_estimators=1000, max_depth=7, min_child_weight=100, gamma=0,
		                     subsample=0.8, colsample_bytree=0.3, objective= "binary:logistic", nthread=4, scale_pos_weight=1)

		xgb1.fit(X_train, y_train)

		y_pred = xgb1.predict(X_test)
		predictions = [round(value) for value in y_pred]
		from sklearn.metrics import accuracy_score

		# evaluate predictions
		accuracy = accuracy_score(y_test, predictions)
		#print("Accuracy: %.2f%%" % (accuracy * 100.0))

		finaldata = {"accuracy": accuracy*100.0}
		results = AccuracySerializer(finaldata).data
		return Response(results)

	def get(self, request):
		yourdata= {"accuracy": 10}
		results = AccuracySerializer(yourdata).data
		return Response(results)
