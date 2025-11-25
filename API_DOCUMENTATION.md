# BranchFusionNet API (v1)

BranchFusionNet provides tomato leaf disease detection using a TensorFlow Lite (TFLite) CNN model (10 classes) plus optional AI (Gemini) advisory output. You upload a leaf image and receive: diagnosis, confidence, structured treatment recommendations, and—if requested—concise AI advice.

Base URL: `https://branchfusionnet.onrender.com`

## Summary of Capabilities
- Image-based disease classification (10 conditions + healthy)
- Structured treatment guidance (immediate / preventive / organic / prognosis)
- Optional AI advice (Gemini) via query flag
- Lightweight TFLite inference (fast, low resource)

## Authentication
None required. All endpoints are public. Use responsibly.

## Core Disease Classes
`Grey_leaf_spot_(fungi)`, `Tomato___Bacterial_spot`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Leaf_Mold`, `Tomato___Septoria_leaf_spot`, `Tomato___Spider_mites`, `Tomato___Target_Spot`, `Tomato___Yellow_Leaf_Curl_Virus`, `Tomato___healthy`.

## Endpoints Overview
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/predict_disease/` | Welcome ping |
| GET | `/predict_disease/health` | Health check |
| GET | `/predict_disease/model/info` | Model metadata + assistant availability |
| POST | `/predict_disease/predict` | Main prediction endpoint |

## Image Requirements
- Format: JPG / JPEG / PNG
- Minimum size: 100x100 px (larger recommended)
- Content: Clear tomato leaf (avoid clutter, blur, harsh glare)
- Lighting: Natural / even; avoid overexposure

## Prediction Endpoint
`POST /predict_disease/predict`

### Request (multipart/form-data)
Field: `file` (required) – image file.

### Optional Query Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `include_advice` | bool | false | If true, attaches AI (Gemini) generated guidance. |
| `region` | string | null | Regional context (e.g. `Nigeria`, `Kenya`). Improves advice relevance. |
| `language` | string | `en` | Language for AI advice (if supported). |

### Example cURL
```bash
curl -X POST "https://branchfusionnet.onrender.com/predict_disease/predict?include_advice=true&region=Nigeria&language=en" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/tomato_leaf.jpg"
```

### Example Python
```python
import requests

url = "https://branchfusionnet.onrender.com/predict_disease/predict"
params = {"include_advice": True, "region": "Nigeria", "language": "en"}
files = {"file": ("leaf.jpg", open("leaf.jpg", "rb"), "image/jpeg")}
res = requests.post(url, params=params, files=files)
data = res.json()

if data.get("success"):
    print("Disease:", data["diagnosis"]["disease"])
    print("Confidence:", f"{data['diagnosis']['confidence']:.2%}")
    print("Immediate treatment:", data["treatment"].get("immediate", [])[:2])
    if "ai_advice" in data and data["ai_advice"].get("available"):
        print("AI Advice:\n", data["ai_advice"]["content"])
else:
    print("Error:", data.get("message"))
```

### Example JavaScript Fetch
```javascript
const form = new FormData();
form.append('file', fileInput.files[0], 'leaf.jpg');

fetch('https://branchfusionnet.onrender.com/predict_disease/predict?include_advice=true&region=Nigeria', {
  method: 'POST',
  body: form
})
  .then(r => r.json())
  .then(d => {
    if (d.success) {
      console.log('Disease:', d.diagnosis.disease);
      console.log('Confidence:', (d.diagnosis.confidence * 100).toFixed(1) + '%');
      console.log('Immediate:', d.treatment.immediate);
      if (d.ai_advice?.available) console.log('AI Advice:', d.ai_advice.content);
    } else {
      console.error('Error:', d.message);
    }
  });
```

## Response Schema (Successful Prediction)
```json
{
  "success": true,
  "diagnosis": {
    "disease": "Tomato___Early_blight",
    "confidence": 0.8945,
    "severity": "moderate"
  },
  "treatment": {
    "immediate": ["Remove lower infected leaves", "Apply fungicide (chlorothalonil, mancozeb)"],
    "preventive": ["Mulch to prevent soil splash", "Water at base"],
    "organic": ["Neem oil spray", "Baking soda solution"]
  },
  "prognosis": "Very treatable when detected early",
  "model_type": "tflite",
  "ai_advice": {
    "available": true,
    "content": "**🚨 Immediate Actions:**\n• Remove infected leaves...",
    "format": "structured"
  }
}
```

### Healthy Example
```json
{
  "success": true,
  "diagnosis": {"disease": "Tomato___healthy", "confidence": 0.9234, "severity": "none"},
  "treatment": {"immediate": ["Continue current care"], "preventive": ["Consistent watering"], "organic": []},
  "prognosis": "Plant is healthy - maintain current practices",
  "model_type": "tflite"
}
```

### Validation Failure Example
```json
{
  "success": false,
  "message": "Please upload a clear image of tomato plant leaves",
  "suggestion": "Ensure the image contains visible green plant material and clear leaf details"
}
```

## Error Codes
| Code | Meaning | Typical Cause |
|------|---------|---------------|
| 200 | OK | Successful prediction |
| 400 | Bad Request | Non-image file / missing file field |
| 404 | Not Found | Wrong endpoint path |
| 500 | Server Error | Internal inference / model load failure |

## Model Info Endpoint
`GET /predict_disease/model/info`
Returns: model type, path, class count, AI assistant availability.

Sample:
```json
{
  "model_type": "tflite",
  "model_path": "model/MultiBranchFusionNet-0.tflite",
  "available_classes": 10,
  "status": "loaded",
  "ai_assistant": "active"
}
```

## AI Advice Notes
- Only returned if `include_advice=true` and Gemini API key configured.
- Structured under `ai_advice` with `available`, `content`, `format`.
- If unavailable: omitted or `{ "available": false, "message": "..." }`.

## Quick Health Script (Batch Folder Scan)
```python
import os, requests

API = "https://branchfusionnet.onrender.com/predict_disease/predict"

def scan(folder):
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, f)
            with open(path, 'rb') as img:
                r = requests.post(API, files={"file": (f, img, "image/jpeg")})
                j = r.json()
                if j.get("success"):
                    d = j["diagnosis"]["disease"]
                    conf = j["diagnosis"]["confidence"]
                    print(f"{f}: {d} ({conf:.1%})")
                else:
                    print(f"{f}: ERROR - {j.get('message')}")

scan("./samples")
```

## Rate Limits
No formal rate limiting presently. Please avoid excessive automated polling.

## Disclaimer
Agronomic output is informational; verify before applying chemical or large-scale interventions. We assume no liability for misuse or decisions made solely on API output.

## Changelog
- v1.0: Initial TFLite prediction service
- v1.1: Added structured treatment recommendations
- v1.2: Added optional Gemini AI advisory (`include_advice` param)

---
For issues: open a GitHub issue or inspect server logs for stack traces.
# BranchFusionNet API v1

BranchFusionNet is a machine learning-based service specifically designed to identify and diagnose tomato plant diseases from leaf images. The API uses a TensorFlow Lite deep learning model trained on 10 distinct disease classes to provide accurate disease classification with confidence scores and expert treatment recommendations.

**Base URL**: `https://branchfusionnet.onrender.com`

## How to use this documentation

This documentation provides comprehensive information about the BranchFusionNet API. The specification of endpoints is provided with detailed examples showing various implementation approaches. You can find example calls in multiple programming languages including Python, JavaScript, Dart/Flutter, and cURL.

## Authentication

Currently, this API does not require authentication. All endpoints are publicly accessible.

## Disease Prediction Request

There are two methods to create a new disease prediction:

**Method 1**: Using `application/json` request - image encoded as Base64 string or public URL in JSON body

**Method 2**: Using `multipart/form-data` request - image sent as a file upload (recommended)

### Request Attributes

**file** - (required) The image file to analyze, provided as:
- `multipart/form-data`: File upload with explicit filename and content type
- `application/json`: Base64-encoded image string or public image URL

### Image Requirements

- **Format**: JPG, JPEG, PNG
- **Size**: Minimum 100x100 pixels recommended
- **Content**: Clear, well-lit tomato plant leaves
- **Quality**: Sufficient detail to identify disease symptoms

## Prediction Result

The response contains comprehensive information about the identified disease. Here are the key components:

**success** - boolean indicating whether the prediction was successful

**disease** - string identifier of the detected condition (e.g., `Tomato___Early_blight` or `Tomato___healthy`)

**confidence** - float (0-1) representing the model's confidence level in the prediction

**disease_info** - object containing detailed information about the identified condition:
- **disease_name** - human-readable name of the disease
- **severity** - disease severity level: `none`, `moderate`, `high`, or `critical`
- **treatment** - comprehensive treatment recommendations object containing:
  - **immediate** - list of urgent actions to take immediately
  - **preventive** - list of measures to prevent future occurrences
  - **organic** - list of chemical-free treatment alternatives
- **prognosis** - expected outcome description with proper treatment

**model_type** - string indicating the model architecture used (returns `tflite`)

## Supported Disease Classes

The API detects 10 distinct conditions affecting tomato plants:

1. **Grey_leaf_spot_(fungi)** - Fungal infection causing grey spots on leaves
2. **Tomato___Bacterial_spot** - Bacterial infection with characteristic spotting
3. **Tomato___Early_blight** - Common fungal disease with concentric ring patterns
4. **Tomato___Late_blight** - Severe fungal disease requiring immediate action
5. **Tomato___Leaf_Mold** - Fungal growth on leaf surfaces
6. **Tomato___Septoria_leaf_spot** - Fungal disease with small circular spots
7. **Tomato___Spider_mites** - Pest infestation causing leaf damage
8. **Tomato___Target_Spot** - Fungal disease with target-like lesions
9. **Tomato___Yellow_Leaf_Curl_Virus** - Viral disease causing leaf curling
10. **Tomato___healthy** - No disease detected; plant is healthy

## Response Codes

| Response Code | Explanation |
|--------------|-------------|
| 200 | Successful request |
| 400 | Invalid input data - file must be a valid image |
| 404 | Endpoint not found |
| 500 | Server error - prediction failed |

⚠️ **Disclaimer**: We are not liable for damages or injury caused by actions based on inaccurate, misleading, incomplete or wrong information provided by this API.

---

## Endpoints

### GET Health Check
`https://branchfusionnet.onrender.com/predict_disease/health`

Checks if the API service is running and accessible.

**Response:**
```json
{
  "status": "ok"
}
```

**Example Request:**
```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/health"
```

---

### GET Welcome Message
`https://branchfusionnet.onrender.com/predict_disease/`

Returns a welcome message from the API service.

**Response:**
```json
{
  "message": "Welcome to the Tomato Plant Disease API service"
}
```

**Example Request:**
```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/"
```

---

### GET Model Information
`https://branchfusionnet.onrender.com/predict_disease/model/info`

Retrieves information about the underlying machine learning model.

**Example Request:**
```bash
curl -X GET "https://branchfusionnet.onrender.com/predict_disease/model/info"
```

---

### POST Disease Prediction (Main Endpoint)
`https://branchfusionnet.onrender.com/predict_disease/predict`

Analyzes uploaded tomato leaf image and returns disease prediction with treatment recommendations

---

## Implementation Examples

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
# Explicitly define filename, file object, and content type
files = {"file": ("tomato_leaf_image.jpg", open("tomato_leaf_image.jpg", "rb"), "image/jpeg")}

response = requests.post(url, files=files)
result = response.json()

if result["success"]:
    print(f"Disease: {result['disease']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Treatment: {result['disease_info']['treatment']['immediate']}")
else:
    print(f"Error: {result['message']}")

print(result)
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
const file = fileInput.files[0];

// Explicitly append the file with the filename to ensure correct processing
formData.append('file', file, "tomato_leaf_image.jpg");

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

### Using Dart/Flutter

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:http_parser/http_parser.dart';

void main() async {
  var uri = Uri.parse('https://branchfusionnet.onrender.com/predict_disease/predict');
  var request = http.MultipartRequest('POST', uri);
  
  // Add file with explicit filename and content type
  var file = await http.MultipartFile.fromPath(
    'file', 
    'tomato_leaf_image.jpg',
    contentType: MediaType('image', 'jpeg'),
  );
  
  request.files.add(file);

  try {
    var streamedResponse = await request.send();
    var response = await http.Response.fromStream(streamedResponse);
    var result = jsonDecode(response.body);

    if (result['success']) {
      print('Disease: ${result['disease']}');
      print('Confidence: ${(result['confidence'] * 100).toStringAsFixed(2)}%');
      print('Treatment: ${result['disease_info']['treatment']['immediate']}');
    } else {
      print('Error: ${result['message']}');
    }
  } catch (e) {
    print('Error: $e');
  }
}
```

---

## Example Response Formats

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

### Error Response (400 Bad Request)

```json
{
  "detail": "File must be an image"
}
```

### Error Response (500 Internal Server Error)

```json
{
  "detail": "Prediction failed: [error details]"
}
```

---

---

## Treatment Information Details

Each disease response includes comprehensive treatment guidance:

- **immediate** - list of strings - urgent actions to take right away to control disease spread
- **preventive** - list of strings - measures to prevent future occurrences and protect healthy plants
- **organic** - list of strings - chemical-free treatment options for organic gardening
- **prognosis** - string - expected outcome description with proper treatment application
- **severity** - string - disease severity classification: `none`, `moderate`, `high`, or `critical`

---

## Rate Limits and Usage

No explicit rate limits are currently enforced. Please use the API responsibly.

---

## Quick Integration Example

```python
import requests
import os

def diagnose_plant(image_path):
    """
    Diagnose tomato plant disease from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        String with diagnosis results or error message
    """
    url = "https://branchfusionnet.onrender.com/predict_disease/predict"
    filename = os.path.basename(image_path)
    
    with open(image_path, 'rb') as f:
        # Explicitly define filename and content type
        files = {"file": (filename, f, "image/jpeg")}
        response = requests.post(url, files=files)
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

---

## Support

For technical issues, check API response error messages for details.