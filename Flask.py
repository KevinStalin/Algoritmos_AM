from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify
#----------------------------------------------------------------
# import Regresion_Logistica as RL
from Control import DATA as D
from Control import Naive_Bayes as NB
from Control import Regresion_Logistica as RL
from Control import Maquina_Soporte_Vectorial as MSV


app = Flask(__name__)

@app.route("/")
def raiz():
    return render_template('index.html',nombresIris=D.nombres_iris)

# def home():
#     return render_template('prueba.html')

@app.route('/ranfore')
def primera():
    
    return render_template('RandomForest.html')

@app.route('/navyes')
def segunda():
    return render_template('NaiveBayes.html',
                           acurracy_I=NB.Acurracy_iris,
                           precision_I=NB.Precision_iris,
                           recall_I=NB.Recall_iris,
                           F1_I=NB.F1_iris,
                           T_errores_I=NB.totalErrores_iris,
                           acurracy_W=NB.Acurracy_wine,
                           precision_W=NB.Precision_wine,
                           recall_W=NB.Recall_wine,
                           F1_W=NB.F1_wine,
                           T_errores_W=NB.totalErrores_wine)

@app.route('/reglog')
def tercera():
    return render_template('RegresionLLogistica.html',
                           acurracy_I=RL.Acurracy_iris,
                           precision_I=RL.Precision_iris,
                           recall_I=RL.Recall_iris,
                           F1_I=RL.F1_iris,
                           T_errores_I=RL.totalErrores_iris,
                           acurracy_W=RL.Acurracy_wine,
                           precision_W=RL.Precision_wine,
                           recall_W=RL.Recall_wine,
                           F1_W=RL.F1_wine,
                           T_errores_W=RL.totalErrores_wine)

@app.route('/redneu')
def cuarta():
    return render_template('RedNeuronal.html')

@app.route('/msv')
def quinta():
    return render_template('SoporteVectorial.html',
                           acurracy_I=MSV.accuracyI,
                           precision_I=MSV.precisionI,
                           recall_I=MSV.recallI,
                           F1_I=MSV.f1I,
                           T_errores_I=MSV.totalErrores_iris,
                           acurracy_W=MSV.accuracyW,
                           precision_W=MSV.precisionW,
                           recall_W=MSV.recallW,
                           F1_W=MSV.f1W,
                           T_errores_W=MSV.totalErrores_wine)

if __name__== '__main__':
    app.run(debug=True)