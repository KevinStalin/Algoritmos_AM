from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('prueba.html')

@app.route('/ranfore')
def primera():
    
    return render_template('RandomForest.html')

@app.route('/navyes')
def segunda():
    return render_template('NaiveBayes.html')

@app.route('/reglog')
def tercera():
    return render_template('RegresionLLogistica.html')

@app.route('/redneu')
def cuarta():
    return render_template('RedNeuronal.html')

@app.route('/msv')
def quinta():
    return render_template('SoporteVectorial.html')



if __name__== '__main__':
    app.run(debug=True)