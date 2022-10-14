from copyreg import pickle
import pickle

# laod the model 

model = pickle.load(open('diabetes_9.pkl',"rb"))

result = model.predict([[1,1,1,1,1,1,1,1]])
if result == 1:
    print("diabetic person")
else:
    print("Non - diabetic person")

