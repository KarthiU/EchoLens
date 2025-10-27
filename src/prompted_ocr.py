import re
from transformers import pipeline
import simple_ocr  # assumes your OCR functions are in simple_ocr.py
from collections import Counter


PROMPTS = [
    "Hey EchoLens, what bus is in front of me?",
    "Hey EchoLens, read and summarize document in front of me",
    "Hey EchoLens, what is that sign in front of me"
]

# Use HuggingFace zero-shot-classification to match user input to prompt
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def match_prompt(user_input):
    result = classifier(user_input, PROMPTS)
    return result['labels'][0]  # best match


def most_common_ocr_bus(num_frames=5): 
    print("Looking for bus number (majority vote)...")
    bus_candidates = []
    for _ in range(num_frames):
        frame = simple_ocr.capture_frame()
        preprocessed_frame = simple_ocr.preprocess_frame_bus(frame)
        ocr_results = simple_ocr.run_ocr_on_frame(preprocessed_frame)
        # Look for a number with 3 digits or less
        for res in ocr_results:
            texts = res.get("rec_texts", [])
            for text in texts:
                match = re.fullmatch(r"\d{1,3}", text)
                if match:
                    bus_candidates.append(text)
        simple_ocr.time.sleep(0.2)  # Small delay between frames

    if bus_candidates:
        most_common = Counter(bus_candidates).most_common(1)[0][0]
        print(f"Majority voted bus number: {most_common}")
        return most_common
    else:
        print("No bus number detected.")
        return None

def run_bus_ocr():
    print("Looking for bus number...")
    found = False
    while not found:
        frame = simple_ocr.capture_frame()
        preprocessed_frame = simple_ocr.preprocess_frame_bus(frame)
        ocr_results = simple_ocr.run_ocr_on_frame(preprocessed_frame)
        # Look for a number with 3 digits or less
        for res in ocr_results:
            texts = res.get("rec_texts", [])
            for text in texts:
                match = re.fullmatch(r"\d{1,3}", text)
                if match:
                    print(f"Bus number detected: {text}")
                    found = True
                    return text
        # Add a small delay to avoid busy loop
        simple_ocr.time.sleep(0.2)

def run_document_ocr():
    print("Reading and summarizing document...")
    frame = simple_ocr.capture_frame()
    
    preprocessed_frame = simple_ocr.preprocess_frame(frame)
    ocr_results = simple_ocr.run_ocr_on_frame(preprocessed_frame)
    all_text = []
    for res in ocr_results:
        all_text.extend(res.get("rec_texts", []))
    document = " ".join(all_text)
    # Use HuggingFace summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(document, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    print(f"Summary: {summary}")
    return summary

def run_sign_ocr():
    print("Reading sign...")
    frame = simple_ocr.capture_frame()
    preprocessed_frame = simple_ocr.preprocess_frame(frame)
    ocr_results = simple_ocr.run_ocr_on_frame(preprocessed_frame)
    sign_texts = []
    for res in ocr_results:
        sign_texts.extend(res.get("rec_texts", []))
    sign_text = " ".join(sign_texts)
    print(f"Sign text: {sign_text}")
    return sign_text

def main():
    # Replace this with microphone input or other user input method
    # user_input = input("Say your prompt: ")
    user_input = "Yo echoLens, what sign is the one im looking at"
    matched = match_prompt(user_input)

    if matched == PROMPTS[0]:
        bus_number = most_common_ocr_bus()
        print(f"Bus: {bus_number}")
    elif matched == PROMPTS[1]:
        summary = run_document_ocr()
        print(f"Document summary: {summary}")
    elif matched == PROMPTS[2]:
        sign = run_sign_ocr()
        print(f"Sign: {sign}")
    else:
        print("Sorry, I didn't understand your request.")

if __name__ == "__main__":
    main()