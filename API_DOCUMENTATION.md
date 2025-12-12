# BranchFusionNet API Documentation

Tomato plant disease detection API using a TensorFlow Lite model with optional AI-powered treatment advice.

**Base URL:** `https://branchfusionnet.onrender.com`

**Version:** 2.0

**Last Updated:** December 12, 2025

---

## Overview

This API provides tomato plant disease detection from leaf images. It uses a two-stage validation process:

1. **Tomato Validation** - Verifies the uploaded image contains a tomato plant
2. **Disease Classification** - Identifies the specific disease (if any) from 10 possible classes

---

## Endpoints

### GET `/predict_disease/`

Welcome endpoint to verify API is running.

**Response:**
```json
{
  "message": "Welcome to the Tomato Plant Disease API service"
}
```

---

### GET `/predict_disease/health`

Health check endpoint for monitoring and keep-alive pings.

**Response:**
```json
{
  "status": "ok"
}
```

---

### GET `/predict_disease/model/info`

Get information about the loaded model and AI assistant status.

**Response:**
```json
{
  "model_type": "tflite",
  "available_classes": 10,
  "status": "loaded",
  "ai_assistant": "active"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | string | Model format: `"tflite"` or `"keras"` |
| `available_classes` | integer | Number of disease classes (always 10) |
| `status` | string | Model load status: `"loaded"` |
| `ai_assistant` | string | AI assistant status: `"active"` or `"inactive"` |

---

### POST `/predict_disease/predict`

Main prediction endpoint. Accepts an image and returns disease diagnosis.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_advice` | boolean | `false` | Include AI-generated treatment advice |
| `region` | string | `null` | Farmer's region for localized advice (e.g., `"Nigeria"`) |
| `language` | string | `"en"` | Response language code (e.g., `"en"`, `"fr"`, `"sw"`) |

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** `file` - Image file (JPEG, PNG, etc.)

---

## Response Formats

The API returns **two different response structures** depending on the validation outcome:

### Response Type 1: Validation Failed

Returned when the image fails validation (not a tomato plant, poor quality, etc.)

```json
{
  "success": false,
  "message": "This does not appear to be a tomato plant image (85.3% confidence).",
  "suggestion": "Please upload a clear image of tomato plant leaves for disease detection."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `false` for validation failures |
| `message` | string | Human-readable explanation with confidence score |
| `suggestion` | string | Actionable guidance for the user |

**Validation Failure Scenarios:**

| Scenario | Example Message |
|----------|-----------------|
| Not a tomato plant | `"This does not appear to be a tomato plant image (85.3% confidence)."` |
| Image too small | `"Image too small (50x50). Minimum 100x100 pixels required."` |
| Insufficient plant material | `"Image lacks sufficient green plant material (5.2%)."` |

---

### Response Type 2: Prediction Success (without advice)

Returned when `include_advice=false` (default), `region=null`, `language=en` and validation passes.

```json
{
  "success": true,
  "confidence": 0.93,
  "disease": "Healthy Plant",
  "treatment": "Ensure current care routine is maintained",
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `true` for successful predictions |
| `confidence` | float | Model prediction confidence score |
| `disease` | string | Disease name |
| `treatment` | string | Structured treatment advice |

**Note:** If AI assistant is inactive, `treatment` will contain a simple advice instead of AI-generated text.

---

### Response Type 3: Prediction Success (with advice)

Returned when `include_advice=true` and validation passes.

```json
{
  "disease": "Healthy Plant",
  "confidence": 1.0,
  "success": true,
  "treatment": "Here are 3 brief tips to maintain your tomato plant's health, tailored for Akure's climate:\n\n1.  **Mindful Watering:** During Akure's dry season, water deeply and consistently. In the rainy season, ensure excellent drainage to prevent waterlogging and root rot from heavy downpours.\n2.  **Vigilant Disease Control:** Akure's high humidity creates ideal conditions for fungal diseases (like blight) and various pests. Inspect plants daily for early signs and treat promptly to prevent widespread issues.\n3.  **Nutrient Replenishment & Airflow:** Heavy rains can leach soil nutrients, so regularly feed your plant with a balanced fertilizer. Prune lower leaves and suckers to improve air circulation, crucial for reducing disease risk in humid Akure conditions."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `true` for successful predictions |
| `disease` | string | Clean disease name (spaces instead of underscores) |
| `confidence` | float | Prediction confidence (0.0 to 1.0) |
| `treatment` | string | AI-generated detailed treatment advice |

---

---

## Supported Disease Classes

| Class ID | Disease Name | Severity |
|----------|--------------|----------|
| 0 | Grey Leaf Spot (Fungal) | Moderate |
| 1 | Bacterial Spot | High |
| 2 | Early Blight | Moderate |
| 3 | Late Blight | Critical |
| 4 | Leaf Mold | Moderate |
| 5 | Septoria Leaf Spot | Moderate |
| 6 | Spider Mites Infestation | Moderate |
| 7 | Target Spot | Moderate |
| 8 | Yellow Leaf Curl Virus | Critical |
| 9 | Healthy Plant | None |

---

## Usage Examples

### cURL

**Basic prediction:**
```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tomato_leaf.jpg"
```

**With AI advice:**
```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict?include_advice=true&region=Nigeria&language=en" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tomato_leaf.jpg"
```

### Python

```python
import requests

url = "https://branchfusionnet.onrender.com/predict_disease/predict"

# Basic prediction
with open("tomato_leaf.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})
    result = response.json()

if result["success"]:
    print(result["disease"])
else:
    print(f"Error: {result['message']}")
    print(f"Suggestions: {result['suggestion']}")
```

```python
# With AI advice
params = {
    "include_advice": True,
    "region": "Nigeria",
    "language": "en"
}

with open("tomato_leaf.jpg", "rb") as f:
    response = requests.post(url, files={"file": f}, params=params)
    result = response.json()

if result["success"]:
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Treatment:\n{result['treatment']}")

else: 
    print(f"Success: {result["success"]}")
    print(f"Message: {result["message"]}")
    print(f"Suggestions: {result["suggestion"]})
```

### JavaScript (Fetch)

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

// Basic prediction
fetch("https://branchfusionnet.onrender.com/predict_disease/predict", {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(data => {
  if (data.success) {
    console.log(data.disease);
    console.log("Treatment:", data.treatment);
  } else {
    console.error(data.message);
    console.log("Suggestion:", data.suggestion);
  }
});
```

```javascript
// With AI advice
const params = new URLSearchParams({
  include_advice: true,
  region: "Nigeria",
  language: "en"
});

fetch(`https://branchfusionnet.onrender.com/predict_disease/predict?${params}`, {
  method: "POST",
  body: formData
})
.then(res => res.json())
.then(data => {
  if (data.success) {
    console.log("Disease:", data.disease);
    console.log("Confidence:", (data.confidence * 100).toFixed(1) + "%");
    console.log("Treatment:", data.treatment);
  }
});
```

---

## Error Responses

### HTTP 400 - Bad Request

```json
{
  "detail": "File must be an image"
}
```

### HTTP 500 - Server Error

```json
{
  "detail": "Model file not found. Please ensure the model is available."
}
```

```json
{
  "detail": "Prediction failed: <error message>"
}
```

---

## Response Decision Tree

```
Upload Image
    │
    ▼
┌─────────────────┐
│ Is Tomato Plant?│
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   No        Yes
    │         │
    ▼         ▼
┌────────┐  ┌─────────────┐
│ FAIL   │  │ Quality OK? │
│Response│  └──────┬──────┘
└────────┘         │
              ┌────┴────┐
              │         │
             No        Yes
              │         │
              ▼         ▼
         ┌────────┐  ┌─────────────┐
         │ FAIL   │  │ Predict     │
         │Response│  │ Disease     │
         └────────┘  └──────┬──────┘
                            │
                    ┌───────┴───────┐
                    │               │
              include_advice   include_advice
                 =false           =true
                    │               │
                    ▼               ▼
              ┌──────────┐   ┌──────────────┐
              │ SUCCESS  │   │ SUCCESS      │
              │ (3 fields)│   │ (6 fields)  │
              └──────────┘   └──────────────┘
```

---

## Best Practices

1. **Image Quality**
   - Use well-lit images of tomato leaves
   - Minimum resolution: 100x100 pixels
   - Ensure leaves are clearly visible and in focus

2. **AI Advice**
   - Enable `include_advice=true` for detailed treatment plans
   - Specify `region` for localized recommendations
   - AI advice requires `GEMINI_API_KEY` to be configured

3. **Error Handling**
   - Always check the `success` field first
   - Display `message` to end users
   - Use `suggestion` to guide user actions

4. **Rate Limiting**
   - No strict rate limits currently enforced
   - Recommended: Max 60 requests/minute for AI advice

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | Dec 12, 2025 | Added tomato validation, simplified response format |
| 1.5 | Dec 10, 2025 | Added Gemini AI assistant integration |
| 1.0 | Dec 1, 2025 | Initial release with TFLite model |

---



---

*This API is for educational and agricultural assistance purposes. Always consult agricultural experts for critical decisions.*
