from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('prueba.html')




if __name__== '__main__':
    app.run(debug=True)

# server_name = app.config['SERVER_NAME']
# if server_name and ':' in server_name:
#     host, port = server_name.split(":")
#     port = int(port)
# else:
#     port = 5000
#     host = "localhost"
# app.run(host=host, port=port)