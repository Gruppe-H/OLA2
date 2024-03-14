from joblib import load
import sys
import json

classifier = load('custom_classifier.joblib')

def predict(data):
    predictions = classifier.predict(data)
    return predictions.tolist()

if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    predictions = predict(input_data)
    print(json.dumps(predictions))