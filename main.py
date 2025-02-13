from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = FastAPI()

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static directory (for CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model & vectorizer file paths
vectorizer_path = "models/tfidf_vectorizer.pkl"  # Updated filename (pickle version)
model_path = "models/best_svm_model.pkl"  # Updated filename (pickle version)

# Ensure both files exist before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"🚨 Vectorizer file '{vectorizer_path}' not found!")

# Load model and vectorizer using Pickle instead of Joblib
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        print(f"✅ Successfully loaded model from '{model_path}'")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        print(f"✅ Successfully loaded vectorizer from '{vectorizer_path}'")

except pickle.UnpicklingError:
    raise RuntimeError("🚨 Error: Model/vectorizer file might be corrupted or saved in an incompatible format!")
except Exception as e:
    raise RuntimeError(f"🚨 Unexpected error while loading model/vectorizer: {e}")

# Ensure correct object types
if not hasattr(model, "predict"):
    raise TypeError("🚨 Error: Loaded model does not have a 'predict' method!")

if not isinstance(vectorizer, TfidfVectorizer):
    raise TypeError("🚨 Error: Loaded vectorizer is not a TfidfVectorizer!")

# Home page (renders the form)
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    """Classify input text using the SVM model"""
    text = text.strip()

    if not text:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Text cannot be empty!",
            "category": None
        })

    try:
        # Transform input text using vectorizer
        transformed_text = vectorizer.transform([text])

        # Debugging: Check transformed text shape
        print(f"🔍 Transformed Text Shape: {transformed_text.shape}")

        # Get prediction
        prediction = model.predict(transformed_text)[0]

        # Debugging: Log prediction result
        print(f"✅ Predicted Category: {prediction}")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "category": str(prediction),
            "error": None
        })

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")  # Log error
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"🚨 Prediction Error: {str(e)}",
            "category": None
        })







# from fastapi import FastAPI, Request, Form
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import joblib
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC

# app = FastAPI()

# # Setup Jinja2 templates
# templates = Jinja2Templates(directory="templates")

# # Mount static directory (for CSS, JS, images)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Define model & vectorizer file paths
# vectorizer_path = "tfidf_vectorizer.pkl"
# model_path = "best_svm_model.pkl"

# # Ensure both files exist before loading
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")

# if not os.path.exists(vectorizer_path):
#     raise FileNotFoundError(f"🚨 Vectorizer file '{vectorizer_path}' not found!")

# # Load model and vectorizer
# model = joblib.load(model_path)
# vectorizer = joblib.load(vectorizer_path)

# # Ensure correct object types
# if not hasattr(model, "predict"):  # Verify model
#     raise TypeError("🚨 Error: Loaded model does not have a 'predict' method!")

# if not isinstance(vectorizer, TfidfVectorizer):  # Verify vectorizer
#     raise TypeError("🚨 Error: Loaded vectorizer is not a TfidfVectorizer!")

# # Home page (renders the form)
# @app.get("/")
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Prediction endpoint
# @app.post("/predict")
# async def predict(request: Request, text: str = Form(...)):
#     if not text.strip():  # Prevent empty input
#         return templates.TemplateResponse("index.html", {"request": request, "error": "Text cannot be empty!"})

#     print(f"🔍 Input text: {text}")  # Debugging print

#     try:
#         transformed_text = vectorizer.transform([text])  # Convert input text
#         print(f"✅ Transformed text shape: {transformed_text.shape}")  # Debugging print

#         prediction = model.predict(transformed_text)[0]  # Get the prediction
#         print(f"🎯 Predicted Category: {prediction}")  # Debugging print

#         return templates.TemplateResponse("index.html", {"request": request, "category": str(prediction)})
    
#     except Exception as e:
#         print(f"⚠️ Prediction error: {e}")  # Debugging print
#         return templates.TemplateResponse("index.html", {"request": request, "error": "Prediction failed!"})
