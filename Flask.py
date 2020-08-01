from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify
#----------------------------------------------------------------
# import Regresion_Logistica as RL
from Control import DATA as D


app = Flask(__name__)
@app.route("/")
def raiz():
    return render_template('index.html',nombresIris=D.nombres_iris)


if __name__== '__main__':
    app.run(debug=True)