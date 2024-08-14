from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    cgpa = float(request.form.get('cgpa'))
    iq = float(request.form.get('iq'))
    scalar = StandardScaler()
    cgpa = (float(cgpa)/10)*100
    iq = (float(iq)/250)*10
    values=np.array([cgpa, iq]).reshape(1,2)
    #values=scalar.fit_transform(values)

    result = model.predict(values)
    return str(result)


if __name__ == '__main__':
    app.run(debug=True)
