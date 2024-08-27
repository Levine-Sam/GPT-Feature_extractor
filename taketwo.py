import openai
import json
import csv
from typing import List, Dict
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def extract_bio_features(bio_text: str) -> Dict:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_bio_info",
                "description": "Extracts structured information from a dating profile bio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of interests mentioned in the bio"
                        },
                        "personality_traits": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Personality traits inferred from the bio"
                        },
                        "hobbies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Hobbies mentioned or implied in the bio"
                        },
                        "career": {
                            "type": "string",
                            "description": "Career or job mentioned in the bio"
                        },
                        "education": {
                            "type": "string",
                            "description": "Educational background mentioned in the bio"
                        },
                        "relationship_goals": {
                            "type": "string",
                            "enum": ["casual", "long-term", "friendship", "not specified"],
                            "description": "Relationship goals mentioned or implied in the bio"
                        },
                        "lifestyle": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Lifestyle choices or preferences mentioned in the bio"
                        },
                        "values": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Personal values or beliefs expressed in the bio"
                        },
                        "communication_style": {
                            "type": "string",
                            "enum": ["direct", "humorous", "sarcastic", "formal", "casual", "not specified"],
                            "description": "Communication style inferred from the bio"
                        },
                        "notable_experiences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Notable life experiences mentioned in the bio"
                        },
                        "bio_sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral", "mixed"],
                            "description": "Overall sentiment of the bio"
                        }
                    },
                    "required": ["interests", "personality_traits", "hobbies", "bio_sentiment"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are an AI assistant that extracts structured information from dating profile bios."},
        {"role": "user", "content": f"Extract key information from this bio according to the provided schema: {bio_text}"}
    ]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            return function_args
        else:
            print("Function call not triggered. The response might be a message instead of structured data.")
            return None
    except openai.RateLimitError as e:
        print(f"Rate limit reached. Retrying in a moment... (Error: {str(e)})")
        time.sleep(random.uniform(1, 5))  
        raise  

def process_bios_from_csv(file_path: str) -> List[Dict]:
    results = []
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            uid, bio = row[0], row[1].strip('*')  
            try:
                extracted_info = extract_bio_features(bio)
                if extracted_info:
                    extracted_info['uid'] = uid
                    results.append(extracted_info)
                    print(f"Successfully processed UID: {uid}")
                else:
                    print(f"Failed to extract information for UID: {uid}")
            except Exception as e:
                print(f"Error processing UID: {uid}. Error: {str(e)}")
                
    return results

def save_results_to_json(results: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_csv = "bio.csv"  
    output_json = "extracted_bio_features.json"
    
    print("Processing bios...")
    extracted_features = process_bios_from_csv(input_csv)
    
    print(f"Saving results to {output_json}...")
    save_results_to_json(extracted_features, output_json)
    
    print(f"Processed {len(extracted_features)} bios. Results saved to {output_json}")
