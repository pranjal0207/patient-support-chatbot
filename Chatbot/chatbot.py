import pickle
import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from spacy.matcher import PhraseMatcher

with open('../Model/disease_model.pkl', 'rb') as f:
    disease_model = pickle.load(f)

with open('../Model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('../Model/drug_recommendation_model.pkl', 'rb') as f:
    drug_model = pickle.load(f)

with open('../Model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

feature_names = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 
                 'Gender_female', 'Gender_male', 
                 'Blood Pressure_high', 'Blood Pressure_low', 'Blood Pressure_normal',
                 'Cholesterol Level_high', 'Cholesterol Level_low', 'Cholesterol Level_normal']

features = [feature.lower() for feature in feature_names]

def recommend_drugs(query):
    data = pd.read_csv("../Data/Processed Data/Cleaned_Drug_Review_Dataset_Train.csv")
    query_vec = tfidf.transform([query])
    
    try:
        distances, indices = drug_model.kneighbors(query_vec, n_neighbors=5)

        valid_indices = [i for i in indices[0] if i < len(data)]
        
        if not valid_indices:
            return "No valid drug recommendations could be found based on your query."

        recommended_drugs = data.iloc[valid_indices]['drugName'].tolist()
        
        if not recommended_drugs:
            return "No valid drug recommendations could be found based on your query."
        
        return recommended_drugs
    
    except ValueError as ve:
        return f"Error in recommendation process: {str(ve)}"
    except IndexError as ie:
        return f"Index error in recommendation process: {str(ie)}"


def predict_disease(symptoms, demographics):
    input_data = np.zeros(len(feature_names))

    symptom_map = {'fever': 0, 'cough': 1, 'fatigue': 2, "tired": 2, 'difficulty breathing': 3, "high blood pressure" : 7, "low blood pressure" : 8, "normal blood pressure" : 9, "low cholestrol" : 11, "high cholestrol" : 10, "normal cholestrol":12}
    for symptom in symptoms:
        if symptom in symptom_map:
            input_data[symptom_map[symptom]] = 1

    
    if demographics.get("Age") is not None:
        input_data[4] = demographics.get("Age")
    
    if demographics.get("Gender") == "female":
        input_data[5] = 1
    elif demographics.get("Gender") == "male":
        input_data[6] = 1

    prediction = disease_model.predict([input_data])
    disease = label_encoder.inverse_transform(prediction)
    
    return disease[0]

def extract_entities(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)

    matcher = PhraseMatcher(nlp.vocab)
    
    symptoms = ["fever", "cough", "tired", "fatigue", "difficulty breathing", "high blood pressure", "low blood pressure", "normal blood pressure", "low cholesterol", "high cholesterol", "normal cholesterol"]
    patterns = [nlp.make_doc(symptom.lower()) for symptom in symptoms]
    matcher.add("SYMPTOM", patterns)
    
    matches = matcher(doc)
    extracted_symptoms = []
    for match_id, start, end in matches:
        extracted_symptoms.append(doc[start:end].text)
    
    demographics = {
        "Age": None,
        "Gender": None
    }
    
    for ent in doc.ents:
        if ent.label_ in ["CARDINAL", "QUANTITY", "DATE"]:
            if "year" in ent.text or ent.text.isdigit() and int(ent.text) < 120:  # Assuming ages < 120
                demographics["Age"] = ent.text
        
        gender_keywords = {"male", "female", "man", "woman", "boy", "girl"}
        for token in doc:
            if token.text.lower() in gender_keywords:
                demographics["Gender"] = token.text.lower()
    
    return extracted_symptoms, demographics


def chat(user_input, context):
    if not user_input:
        return {"response": "I'm sorry, I didn't understand that."}

    symptoms, demographics = extract_entities(user_input)
    context["symptoms"].extend(symptoms)
    context["demographics"].update(demographics)

    if "diagnose" in user_input.lower() or "symptoms" in user_input.lower():
        disease = predict_disease(context["symptoms"], context["demographics"])
        response = f"Based on the symptoms you provided, you may have {disease}. Please consult with a healthcare provider for a professional diagnosis."

    elif "recommend" in user_input.lower() or "drug" in user_input.lower():
        drugs = recommend_drugs(user_input)
        response = f"Here are some drugs that may be helpful: {', '.join(drugs)}."

    else:
        response = "Could you please specify whether you need a diagnosis or a drug recommendation?"

    return {"response": response}

if __name__ == "__main__":
    context = {
        "symptoms": [],
        "demographics": {},
        "condition": None,
    }

    user_input = "Can you recommend something for high blood pressure?"
    result = chat(user_input, context)
    print(result["response"])