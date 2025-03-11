import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define our symptom features (input)
# Fever, Cough, Fatigue, Headache, Sore Throat
symptoms = np.array([
    [1, 0, 1, 0, 0],  # Patient 1
    [0, 1, 0, 0, 1],  # Patient 2
    [1, 1, 1, 0, 0],  # Patient 3
    [0, 0, 1, 1, 0],  # Patient 4
    [1, 1, 0, 0, 1],  # Patient 5
])

# Define diagnosis labels (output)
# Cold, Flu, COVID, Allergy
diagnoses = np.array([
    [0, 1, 0, 0],  # Patient 1 - Flu
    [1, 0, 0, 0],  # Patient 2 - Cold
    [0, 0, 1, 0],  # Patient 3 - COVID
    [0, 0, 0, 1],  # Patient 4 - Allergy
    [1, 0, 0, 0],  # Patient 5 - Cold
])

# Create a simple neural network
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(5,)),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(symptoms, diagnoses, epochs=100, verbose=0)

# Model can now predict diagnoses based on symptoms

class SymbolicReasoner:
    def __init__(self):
        # Define medical knowledge as rules
        self.rules = [
            # Rule format: (diagnosis, required_symptoms, contradicting_symptoms, explanation)
            ("Cold", ["Cough", "Sore Throat"], ["High Fever"], 
             "Patient has Cold symptoms: cough and sore throat without high fever"),
            
            ("Flu", ["Fever", "Fatigue"], [], 
             "Patient has Flu symptoms: fever and fatigue"),
            
            ("COVID", ["Fever", "Cough", "Fatigue"], [], 
             "Patient has COVID symptoms: fever, cough and fatigue"),
            
            ("Allergy", ["Headache"], ["Fever"], 
             "Patient has Allergy symptoms: headache without fever")
        ]
        
        # Mapping from symptom index to name
        self.symptom_names = ["Fever", "Cough", "Fatigue", "Headache", "Sore Throat"]
        self.diagnosis_names = ["Cold", "Flu", "COVID", "Allergy"]
    
    def get_patient_symptoms(self, symptom_vector):
        """Convert symptom vector to list of symptom names"""
        return [self.symptom_names[i] for i, has_symptom in enumerate(symptom_vector) if has_symptom]
    
    def verify_diagnosis(self, diagnosis_index, symptom_vector):
        """Verify if diagnosis is consistent with symbolic rules"""
        diagnosis = self.diagnosis_names[diagnosis_index]
        patient_symptoms = self.get_patient_symptoms(symptom_vector)
        
        for rule_diagnosis, required, contradicting, explanation in self.rules:
            if rule_diagnosis == diagnosis:
                # Check if all required symptoms are present
                if all(symptom in patient_symptoms for symptom in required):
                    # Check if no contradicting symptoms are present
                    if not any(symptom in patient_symptoms for symptom in contradicting):
                        return True, explanation
        
        return False, "Diagnosis does not match medical knowledge rules"

class NeuralSymbolicSystem:
    def __init__(self, neural_model, symbolic_reasoner):
        self.neural_model = neural_model
        self.symbolic_reasoner = symbolic_reasoner
    
    def diagnose(self, symptoms):
        # Step 1: Get neural network prediction
        neural_prediction = self.neural_model.predict(np.array([symptoms]))[0]
        predicted_diagnosis = np.argmax(neural_prediction)
        confidence = neural_prediction[predicted_diagnosis]
        
        # Step 2: Verify with symbolic reasoner
        is_valid, explanation = self.symbolic_reasoner.verify_diagnosis(
            predicted_diagnosis, symptoms)
        
        # Step 3: Final decision with explanation
        if is_valid:
            final_diagnosis = self.symbolic_reasoner.diagnosis_names[predicted_diagnosis]
            return {
                "diagnosis": final_diagnosis,
                "confidence": float(confidence),
                "explanation": explanation,
                "verified": True
            }
        else:
            # Find alternative diagnosis using symbolic rules
            for i, diagnosis in enumerate(self.symbolic_reasoner.diagnosis_names):
                is_valid, explanation = self.symbolic_reasoner.verify_diagnosis(i, symptoms)
                if is_valid:
                    return {
                        "diagnosis": diagnosis,
                        "confidence": float(neural_prediction[i]),
                        "explanation": explanation + " (neural prediction overridden)",
                        "verified": True
                    }
            
            # If no valid diagnosis found
            return {
                "diagnosis": "Uncertain",
                "confidence": 0.0,
                "explanation": "No consistent diagnosis found",
                "verified": False
            }

# Initialize our hybrid system
reasoner = SymbolicReasoner()
hybrid_system = NeuralSymbolicSystem(model, reasoner)

# Test cases
test_cases = [
    # Fever, Cough, Fatigue, Headache, Sore Throat
    [1, 0, 1, 0, 0],  # Classic flu symptoms
    [0, 1, 0, 0, 1],  # Classic cold symptoms
    [1, 1, 1, 0, 0],  # COVID symptoms
    [0, 0, 0, 1, 1],  # Unusual combination
    [1, 0, 0, 1, 0],  # Fever and headache
]

for i, symptoms in enumerate(test_cases):
    result = hybrid_system.diagnose(symptoms)
    print(f"\nPatient {i+1} with symptoms: {reasoner.get_patient_symptoms(symptoms)}")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Explanation: {result['explanation']}")
    print(f"Verified: {result['verified']}")