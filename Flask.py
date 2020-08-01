from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify
# ----------------------------------------------------------------
# import Regresion_Logistica as RL
# from Control import DATA as D

from Control import Naive_Bayes as NB
from Control import Regresion_Logistica as RL
from Control import KNN as KNN
from Control import Maquina_Soporte_Vectorial as MSV

# from Control import Maquina_Soporte_Vectorial as MSV


app = Flask(__name__)


@app.route("/")
def raiz():
    return render_template('prueba.html',
                           nombres_I=KNN.nombres_iris,
                           acurracy_I=KNN.Acurracy_iris,
                           precision_I=KNN.Precision_iris,
                           recall_I=KNN.Recall_iris,
                           F1_I=KNN.F1_iris,
                           T_errores_I=KNN.totalErrores_iris,
                           nombres_W=KNN.nombres_wine,
                           acurracy_W=KNN.Acurracy_wine,
                           precision_W=KNN.Precision_wine,
                           recall_W=KNN.Recall_wine,
                           F1_W=KNN.F1_wine,
                           T_errores_W=KNN.totalErrores_wine,
                           training_I=KNN.Training_I,
                           test_I=KNN.Test_I,
                           training_W=KNN.Training_W,
                           test_W=KNN.Test_W,
                           data_I=KNN.X_iris,
                           data_W=KNN.X_wine,
                           labels_I=KNN.labels_I,
                           labels_W=KNN.labels_W)

# def home():
#     return render_template('prueba.html')


@app.route('/ranfore')
def primera():
    return render_template('RandomForest.html')

@app.route('/navyes')
def segunda():
    return render_template('NaiveBayes.html',
                           nombres_I=NB.nombres_iris,
                           acurracy_I=NB.Acurracy_iris,
                           precision_I=NB.Precision_iris,
                           recall_I=NB.Recall_iris,
                           F1_I=NB.F1_iris,
                           T_errores_I=NB.totalErrores_iris,
                           nombres_W=NB.nombres_wine,
                           acurracy_W=NB.Acurracy_wine,
                           precision_W=NB.Precision_wine,
                           recall_W=NB.Recall_wine,
                           F1_W=NB.F1_wine,
                           T_errores_W=NB.totalErrores_wine,
                           training_I=NB.Training_I,
                           test_I=NB.Test_I,
                           training_W=NB.Training_W,
                           test_W=NB.Test_W,
                           data_I=NB.X_iris,
                           data_W=NB.X_wine,
                           labels_I=NB.labels_I,
                           labels_W=NB.labels_W)


@app.route('/reglog')
def tercera():
    return render_template('RegresionLLogistica.html',
                           nombres_I=RL.nombres_iris,
                           acurracy_I=RL.Acurracy_iris,
                           precision_I=RL.Precision_iris,
                           recall_I=RL.Recall_iris,
                           F1_I=RL.F1_iris,
                           T_errores_I=RL.totalErrores_iris,
                           nombres_W=RL.nombres_wine,
                           acurracy_W=RL.Acurracy_wine,
                           precision_W=RL.Precision_wine,
                           recall_W=RL.Recall_wine,
                           F1_W=RL.F1_wine,
                           T_errores_W=RL.totalErrores_wine,
                           training_I=RL.Training_I,
                           test_I=RL.Test_I,
                           training_W=RL.Training_W,
                           test_W=RL.Test_W,
                           data_I=RL.X_iris,
                           data_W=RL.X_wine,
                           labels_I=RL.labels_I,
                           labels_W=RL.labels_W)


@app.route('/redneu')
def cuarta():
    return render_template('RedNeuronal.html')

@app.route('/msv')
def quinta():
    return render_template('SoporteVectorial.html',
                           nombres_I=MSV.nombres_iris,
                           acurracy_I=MSV.Acurracy_iris,
                           precision_I=MSV.Precision_iris,
                           recall_I=MSV.Recall_iris,
                           F1_I=MSV.F1_iris,
                           T_errores_I=MSV.totalErrores_iris,
                           nombres_W=MSV.nombres_wine,
                           acurracy_W=MSV.Acurracy_wine,
                           precision_W=MSV.Precision_wine,
                           recall_W=MSV.Recall_wine,
                           F1_W=MSV.F1_wine,
                           T_errores_W=MSV.totalErrores_wine,
                           training_I=MSV.Training_I,
                           test_I=MSV.Test_I,
                           training_W=MSV.Training_W,
                           test_W=MSV.Test_W,
                           data_I=MSV.X_iris,
                           data_W=MSV.X_wine,
                           labels_I=MSV.labels_I,
                           labels_W=MSV.labels_W)

if __name__ == '__main__':
    app.run(debug=True)
