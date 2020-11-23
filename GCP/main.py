# -*- coding: utf-8 -*-
"""
Bereitstellung eines Ml-Modells mittels Flask
"""

# Libraries laden
import numpy as np
from flask import Flask, request, jsonify
import pickle

# App initialisieren
app = Flask(__name__)

# Modell laden
model = pickle.load(open("model.pkl", "rb"))

# Vorbereitung für einfache Validierung des Inputs
error_message='''   
Error - wrong request format - The request needs to be in the format (Values given as examples): 
{"tenure":2, "TotalCharges":1000, "PhoneService_Yes":1, "InternetService_Fiber optic":0, 
 "InternetService_No":1, "Contract_Month-to-Month":1, "Contract_Two year":0, "PaymentMethod_Electronic check":0}
'''
varnames = ["tenure", "InternetService_Fiber optic", "InternetService_No", 
            "TechSupport_No internet service", "StreamingTV_Yes", "Contract_Month-to-month", 
            "Contract_Two year", "MonthlyCharges", "TotalCharges", "OnlineSecurity_No", "TechSupport_No"]


# API Schnittstelle konfigurieren
@app.route("/", methods=["POST"]) # Hier der gewünschte Url-Anhang
# Wir haben nur eine Funktionalität also bleiben wir bei home mit "/"
# POST Methode wählen für maschinelle Interaktion
def home(): # auszuführende Funktion definieren
    # Daten einlesen:
    data = request.get_json(force=True)
    # Modell Vorhersage erstellen:
    if list(data.keys()) == varnames: # Input Validierung
        prediction = model.predict([np.array(list(data.values()))]) # Vorhersage
        output = prediction[0] # Vorhersage Ergebnis speichern
        # Vorhersage als JSON verpacken und als Antwort zurücksenden
        return jsonify(int(output)) # Ergebnis zurücksenden
    else:
        return error_message # Bei fehlerhaftem Ergebnis error_message ausgeben

# App starten
if __name__ == "__main__":
    app.run(debug=False)
