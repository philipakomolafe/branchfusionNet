# BranchFusionNet API

Tomato plant disease detection API using a TensorFlow Lite model with optional AI-powered treatment advice.

**Base URL:** `https://branchfusionnet.onrender.com`

---

## Endpoints

### GET `/predict_disease/`

Welcome message.

**Response:**
```json
{
  "message": "Welcome to the Tomato Plant Disease API service"
}
```

---

### GET `/predict_disease/health`

Health check.

**Response:**
```json
{
  "status": "ok"
}
```

---

### GET `/predict_disease/model/info`

Get model information.

**Response:**
```json
{
  "model_type": "tflite",
  "available_classes": 10,
  "status": "loaded",
  "ai_assistant": "active"
}
```

---

### POST `/predict_disease/predict`

Predict tomato plant disease from uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (required) - image file

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_advice` | bool | `false` | Include AI-generated treatment advice |
| `region` | string | `null` | Farmer's region for tailored advice |
| `language` | string | `"en"` | Response language |

---

## Usage Examples

### cURL

```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict" \
  -F "file=@tomato_leaf.jpg"
```

With AI advice:
```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict?include_advice=true&region=Nigeria&language=en" \
  -F "file=@tomato_leaf.jpg"
```

### Python

```python
import requests

url = "https://branchfusionnet.onrender.com/predict_disease/predict"
files = {"file": ("leaf.jpg", open("leaf.jpg", "rb"), "image/jpeg")}
params = {"include_advice": True, "region": "Nigeria", "language": "en"}

response = requests.post(url, files=files, params=params)
result = response.json()

if result["success"]:
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    if "treatment" in result:
        print(f"Treatment: {result['treatment']}")
else:
    print(f"Error: {result['message']}")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://branchfusionnet.onrender.com/predict_disease/predict?include_advice=true&region=Nigeria', {
  method: 'POST',
  body: formData
})
.then(r => r.json())
.then(data => {
  if (data.success) {
    console.log('Disease:', data.disease);
    console.log('Confidence:', data.confidence);
    if (data.treatment) console.log('Treatment:', data.treatment);
  } else {
    console.error('Error:', data.message);
  }
});
```

---

## Response Examples

### Successful Prediction (without `include_advice`)

```json
{
  "success": true,
  "disease": "Tomato___Early_blight",
  "confidence": 0.8945
}
```

### Successful Prediction (with `include_advice=true`, AI active)

```json
{
  "success": true,
  "disease": "Early blight",
  "confidence": 0.8945,
  "treatment": "1. Immediate Action: Remove infected leaves...\n2. Organic Solution: Apply neem oil...\n3. Prevention: Practice crop rotation..."
}
```

### Successful Prediction (with `include_advice=true`, AI inactive)

Falls back to predefined treatment data:

```json
{
  "success": true,
  "disease": "Early blight",
  "confidence": 0.8945,
  "treatment": {
    "disease_name": "Early Blight",
    "severity": "moderate",
    "treatment": {
      "immediate": [
        "Remove lower infected leaves",
        "Apply fungicide (chlorothalonil, mancozeb)",
        "Stake plants to improve air circulation"
      ],
      "preventive": [
        "Mulch around plants to prevent soil splash",
        "Water at base of plants",
        "Prune lower branches",
        "Practice crop rotation"
      ],
      "organic": [
        "Use baking soda spray",
        "Apply neem oil regularly",
        "Use copper fungicide",
        "Compost tea applications"
      ]
    },
    "prognosis": "Very treatable when detected early"
  }
}
```

### Healthy Plant

```json
{
  "success": true,
  "disease": "Tomato___healthy",
  "confidence": 0.9234
}
```

### Failed Validation

```json
{
  "success": false,
  "message": "Please upload a clear image of tomato plant leaves",
  "suggestion": "Ensure the image contains visible green plant material and clear leaf details"
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "File must be an image"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Model file not found. Please ensure the model is available."
}
```

```json
{
  "detail": "Prediction failed: [error details]"
}
```

---

## Supported Disease Classes

1. `Grey_leaf_spot_(fungi)`
2. `Tomato___Bacterial_spot`
3. `Tomato___Early_blight`
4. `Tomato___Late_blight`
5. `Tomato___Leaf_Mold`
6. `Tomato___Septoria_leaf_spot`
7. `Tomato___Spider_mites`
8. `Tomato___Target_Spot`
9. `Tomato___Yellow_Leaf_Curl_Virus`
10. `Tomato___healthy`

---

## Image Requirements

- Format: JPG, JPEG, PNG
- Minimum size: 100x100 pixels
- Content: Clear tomato plant leaves
- Must contain visible green plant material

---

## Notes

- When `include_advice=true` and AI assistant is active, `treatment` contains AI-generated text advice.
- When `include_advice=true` but AI assistant is inactive, `treatment` contains predefined structured treatment data.
- Without `include_advice`, response only contains `disease` and `confidence`.
- Disease name is cleaned (removes `Tomato___` prefix) when `include_advice=true`.
