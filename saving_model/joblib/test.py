import joblib

# laod the model 

model = joblib.load('diabetes_9.pkl')

result = model.predict([[1,1,1,1,1,1,1,1]])
if result == 1:
    print("diabetic person")
else:
    print("Non - diabetic person")

