from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = FastAPI()

#BASE_DIR = os.path.dirname(os.path.abdpath())
# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Mount static directory (for CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define model & vectorizer file paths
vectorizer_path = "models/tfidf_vectorizer.pkl"  # Changed to store in a "models" folder
model_path = "models/best_svm_model.pkl"

# Ensure both files exist before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 Model file '{model_path}' not found!")

if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"🚨 Vectorizer file '{vectorizer_path}' not found!")

# Load model and vectorizer
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"🚨 Error loading model/vectorizer: {e}")

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
    """Classify news article"""
    text = text.strip()

    if not text:  # Prevent empty input
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Text cannot be empty!",
            "category": None  # Ensure previous result clears
        })

    try:
        transformed_text = vectorizer.transform([text])  # Convert input text

        # Debugging print: Check transformed text
        print(f"🔍 Transformed Text Shape: {transformed_text.shape}")

        prediction = model.predict(transformed_text)[0]  # Get the prediction

        # Debugging print: Check predicted category
        print(f"✅ Predicted Category: {prediction}")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "category": str(prediction),
            "error": None  # Ensure previous errors clear
        })

    except Exception as e:
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
