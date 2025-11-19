# Tomato Plant Disease Detection API

## Overview
This API provides tomato plant disease classification using a TensorFlow Lite deep learning model. Upload an image of tomato plant leaves to get disease predictions with confidence scores and treatment recommendations.

**Base URL**: `https://branchfusionnet.onrender.com`

## Endpoints

### 1. Health Check
**GET** `/predict_disease/health`

```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/health"
```

**Response:**
```json
{
  "status": "ok"
}
```

### 2. Welcome Message
**GET** `/predict_disease/`

```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/"
```

**Response:**
```json
{
  "message": "Welcome to the Tomato Plant Disease API service"
}
```

### 3. Disease Prediction (Main Endpoint)
**POST** `/predict_disease/predict`

Upload an image to get disease prediction with treatment recommendations.

### 4. Model Information
**GET** `/predict_disease/model/info`

```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/model/info"
```

## How to Use the Prediction API

### Using cURL

```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/tomato_leaf_image.jpg"
```

### Using Python Requests

```python
import requests

url = "https://branchfusionnet.onrender.com/predict_disease/predict"
files = {"file": open("tomato_leaf_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

if result["success"]:
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Treatment: {result['disease_info']['treatment']['immediate']}")
else:
    print(f"Error: {result['message']}")
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('https://branchfusionnet.onrender.com/predict_disease/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    console.log('Disease:', data.disease);
    console.log('Confidence:', data.confidence);
    console.log('Treatment:', data.disease_info.treatment);
  } else {
    console.error('Error:', data.message);
  }
});
```

### Using Postman

1. Set method to **POST**
2. URL: `https://branchfusionnet.onrender.com/predict_disease/predict`
3. Body → form-data
4. Key: `file` (type: File)
5. Select your image file
6. Send

## Response Format

### Successful Disease Detection

```json
{
  "success": true,
  "disease": "Tomato___Early_blight",
  "confidence": 0.8945,
  "disease_info": {
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
  },
  "model_type": "tflite"
}
```

### Healthy Plant Detection

```json
{
  "success": true,
  "disease": "Tomato___healthy",
  "confidence": 0.9234,
  "disease_info": {
    "disease_name": "Healthy Plant",
    "severity": "none",
    "treatment": {
      "immediate": [
        "Continue current care routine",
        "Monitor regularly for early signs of disease"
      ],
      "preventive": [
        "Maintain consistent watering",
        "Ensure proper nutrition",
        "Monitor for pests regularly"
      ]
    },
    "prognosis": "Plant is healthy - maintain current practices"
  },
  "model_type": "tflite"
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

## Supported Disease Classes

The API detects 10 conditions:

1. **Grey_leaf_spot_(fungi)** - Grey leaf spot (fungal)
2. **Tomato___Bacterial_spot** - Bacterial spot 
3. **Tomato___Early_blight** - Early blight
4. **Tomato___Late_blight** - Late blight
5. **Tomato___Leaf_Mold** - Leaf mold
6. **Tomato___Septoria_leaf_spot** - Septoria leaf spot
7. **Tomato___Spider_mites** - Spider mites
8. **Tomato___Target_Spot** - Target spot
9. **Tomato___Yellow_Leaf_Curl_Virus** - Yellow leaf curl virus
10. **Tomato___healthy** - Healthy plant

## Treatment Information

Each disease response includes:

- **Immediate actions** - What to do right away
- **Preventive measures** - How to prevent future occurrences  
- **Organic treatments** - Chemical-free treatment options
- **Prognosis** - Expected outcome with proper treatment
- **Severity level** - none/moderate/high/critical

## Image Requirements

- **Format**: JPG, JPEG, PNG
- **Size**: Minimum 100x100 pixels
- **Content**: Clear tomato plant leaves
- **Quality**: Well-lit with visible details

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
  "detail": "Prediction failed: [error details]"
}
```

## Quick Integration Example

```python
import requests

def diagnose_plant(image_path):
    url = "https://branchfusionnet.onrender.com/predict_disease/predict"
    
    with open(image_path, 'rb') as f:
        response = requests.post(url, files={"file": f})
        result = response.json()
    
    if result["success"]:
        if result["disease"] == "Tomato___healthy":
            return "✅ Plant is healthy!"
        else:
            disease_name = result["disease_info"]["disease_name"]
            confidence = result["confidence"]
            immediate_actions = result["disease_info"]["treatment"]["immediate"]
            
            return f"⚠️ {disease_name} detected ({confidence:.1%} confidence)\n" + \
                   f"Immediate actions: {', '.join(immediate_actions[:2])}"
    else:
        return f"❌ {result['message']}"

# Usage
print(diagnose_plant("leaf_image.jpg"))
```

## Rate Limits

No explicit rate limits. Please use responsibly.

## Support

For technical issues, check API response error messages for details.