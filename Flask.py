from flask import Flask, render_template
from flask import make_response
from flask import request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return "hi"
