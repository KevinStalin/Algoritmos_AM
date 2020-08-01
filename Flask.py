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
from Control import Random_Forest as RF
from Control import Red_Neuronal as RN


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
    return render_template('RandomForest.html',
                           nombres_I=RF.features_iris,
                           acurracy_I=RF.accuracy,
                           precision_I=RF.precision,
                           recall_I=RF.recall,
                           F1_I=RF.f1,
                           T_errores_I=RF.totalErrores_iris,
                           nombres_W=RF.features_wine,
                           acurracy_W=RF.accuracy2,
                           precision_W=RF.precision2,
                           recall_W=RF.recall2,
                           F1_W=RF.f1_2,
                           T_errores_W=RF.totalErrores_wine,
                           training_I=RF.train_iris,
                           test_I=RF.test_iris,
                           training_W=RF.train_wine,
                           test_W=RF.test_wine,
                           data_I=KNN.X_iris,
                           data_W=KNN.X_wine,
                           labels_I=RF.labels_Iris,
                           labels_W=RF.labels_Wine)



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
    return render_template('RedNeuronal.html',
                           nombres_I=RN.nombres_iris,
                           acurracy_I=RN.accuracyi,
                           precision_I=RN.precisioni,
                           recall_I=RN.recalli,
                           F1_I=RN.f1i,
                           T_errores_I=RN.totalErrores_iris,
                           nombres_W=RN.nombres_wine,
                           acurracy_W=RN.accuracyw,
                           precision_W=RN.precisionw,
                           recall_W=RN.recallw,
                           F1_W=RN.f1w,
                           T_errores_W=RN.totalErrores_wine,
                           training_I=RN.Training_iris,
                           test_I=RN.Test_iris,
                           training_W=RN.Training_Wine,
                           test_W=RN.Test_Wine,
                           data_I=NB.X_iris,
                           data_W=NB.X_wine,
                           labels_I=RN.labels_I,
                           labels_W=RN.labels_W)
                          



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
                         
@app.route('/bench')
def sexta():
    return render_template('benchmarking.html',

                            acurracy_I=KNN.Acurracy_iris,
                            T_errores_I=KNN.totalErrores_iris,
                            test_I=KNN.Test_I,
                            acurracy_W=KNN.Acurracy_wine,
                            T_errores_W=KNN.totalErrores_wine,
                            test_W=KNN.Test_W,
            #//////////////////////////randon/////////////////////
                            acurracy_I1=RF.accuracy,
                            T_errores_I1=RF.totalErrores_iris,
                            test_I1=RF.test_iris,
                            acurracy_W1=RF.accuracy2,
                            T_errores_W1=RF.totalErrores_wine,
                            test_W1=RF.test_wine,     

            #/////////////////////////////nedvayes//////////////////
                            acurracy_I2=NB.Acurracy_iris,
                            T_errores_I2=NB.totalErrores_iris,
                            test_I2=NB.Test_I,
                            acurracy_W2=NB.Acurracy_wine,
                            T_errores_W2=NB.totalErrores_wine,
                            test_W2=NB.Test_W,
            #///////////////////////////regresion lofistica////////////////////
                            acurracy_I3=RL.Acurracy_iris,
                            T_errores_I3=RL.totalErrores_iris,
                            test_I3=RL.Test_I,
                            acurracy_W3=RL.Acurracy_wine,
                            T_errores_W3=RL.totalErrores_wine,
                            test_W3=RL.Test_W,

            #///////////////////////////red neuronal////////////////////
                            acurracy_I4=RN.accuracyi,
                            T_errores_I4=RN.totalErrores_iris,
                            test_I4=RN.Test_iris,
                            acurracy_W4=RN.accuracyw,
                            T_errores_W4=RN.totalErrores_wine,
                            test_W4=RN.Test_Wine,

            #///////////////////////////soporte vectorial////////////////////
                            acurracy_I5=MSV.Acurracy_iris,
                            T_errores_I5=MSV.totalErrores_iris,
                            test_I5=MSV.Test_I,
                            acurracy_W5=MSV.Acurracy_wine,
                            T_errores_W5=MSV.totalErrores_wine,
                            test_W5=MSV.Test_W


    )


if __name__ == '__main__':
    app.run(debug=True)
