import boto3
import json
import io
import PyPDF2  # Install using: pip install PyPDF2
import requests  # Required for Llama 3 API requests

# Initialize AWS Clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")

# Llama 3 Endpoint (Modify based on your setup)
LLAMA3_API_URL = "http://localhost:11434/api/generate"  # Example for Ollama
LLAMA3_MODEL_NAME = "llama3"  # Adjust if using a different version


# 🔹 Fetch & Extract Text from PDF in S3
def read_s3_pdf(bucket_name, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_stream = io.BytesIO(response["Body"].read())

        pdf_reader = PyPDF2.PdfReader(file_stream)
        text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        return text if text else "No text found in PDF."
    except Exception as e:
        print(f"❌ Error reading S3 PDF: {e}")
        return None


# 🔹 Invoke Bedrock Model with PDF Data
def invoke_bedrock_model(model_id, prompt_text):
    request_body = json.dumps({
        "inputText": prompt_text,
        "textGenerationConfig": {
            "maxTokenCount": 300,
            "temperature": 0.7,
            "topP": 0.9
        }
    })

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=request_body
        )
        response_body = response["body"].read().decode("utf-8")
        return json.loads(response_body)
    
    except Exception as e:
        print(f"❌ Error invoking Bedrock Model {model_id}: {str(e)}")
        return None


# 🔹 Invoke Llama 3 Model (Locally or API)
def invoke_llama3_model(prompt_text):
    try:
        request_data = {"model": LLAMA3_MODEL_NAME, "prompt": prompt_text, "stream": False}
        response = requests.post(LLAMA3_API_URL, json=request_data)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"❌ Llama 3 API Error: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error invoking Llama 3 Model: {e}")
        return None


# 🔹 Specify S3 Bucket & File
bucket_name = "nityombaalti"
file_key = "sem.pdf"

# Read PDF content from S3
s3_data = read_s3_pdf(bucket_name, file_key)

if s3_data:
    prompt_text = f"Based on this document: {s3_data[:500]}\n\nAnswer this: Humam Computer Interaction unit ?" 

    # Invoke Bedrock Titan Model
    response_bedrock = invoke_bedrock_model("amazon.titan-text-express-v1", prompt_text)

    # Invoke Llama 3 Model
    response_llama3 = invoke_llama3_model(prompt_text)

    # ✅ Print the AI-generated response
    print("\n📝 AI Responses:")

    if response_bedrock:
        print("\n🔹 Bedrock Response:\n" + response_bedrock["results"][0]["outputText"])
    else:
        print("\n❌ No response from Bedrock.")

    if response_llama3:
        print("\n🔹 Llama 3 Response:\n" + response_llama3)
    else:
        print("\n❌ No response from Llama 3.")

else:
    print("\n❌ Could not fetch or process PDF from S3.")
