
from flask import Flask,render_template,request,url_for,redirect
from pymongo import MongoClient 

app = Flask(__name__)
client = MongoClient("mongodb+srv://admin-asphy:Akashanimelord18@cluster0.ohbxwdy.mongodb.net/")
db = client['Gknowme2']
col = db['finals']




