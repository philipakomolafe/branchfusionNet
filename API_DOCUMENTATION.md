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