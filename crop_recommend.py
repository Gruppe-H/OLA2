import pickle

def load_model(filename='crop_recommend.pkl'):
    """Load the trained model from a file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_input_features():
    """Prompt the user to input the values for the model's features."""
    print("Enter the values for the following parameters:")
    nitrogen = float(input("Nitrogen: "))
    phosphorus = float(input("Phosphorus: "))
    potassium = float(input("Potassium: "))
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    ph = float(input("PH: "))
    rainfall = float(input("Rainfall: "))
    return [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]

def main():
    model = load_model()

    # Define the names of the crops that your model predicts
    crops = ['Crop1', 'Crop2', 'Crop3', 'Crop4', '...']  # Update this with actual crop names

    data_to_predict = [get_input_features()]

    # Make predictions with probabilities
    probabilities = model.predict_proba(data_to_predict)

    print("\nCrop Suitability Scores:")
    for crop, probability in zip(crops, probabilities[0]):
        print(f"{crop}: {probability:.2%}")
        
    predictions = model.predict(data_to_predict)
    print(f"\nCrop with highest score: {predictions[0]}")
    


if __name__ == '__main__':
    main()
