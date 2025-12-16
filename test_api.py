"""Test script for the Careerflow API."""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api"

def test_full_flow():
    # Upload resume first
    print("=== UPLOADING RESUME ===")
    with open('test_resume.docx', 'rb') as f:
        response = requests.post(
            f'{BASE_URL}/resume/upload',
            files={'file': ('test_resume.docx', f, 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
        )
        upload_data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Resume ID: {upload_data.get('resume_id')}")
        print(f"Sections: {upload_data.get('sections_detected')}")
        resume_id = upload_data.get('resume_id')

    # Test optimize for Google
    print("\n=== OPTIMIZE FOR GOOGLE ===")
    response = requests.post(
        f'{BASE_URL}/chat/message',
        json={
            'message': 'Optimize this resume for Google',
            'resume_id': resume_id
        },
        timeout=120
    )
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Agent: {data.get('agent_type')}")
    print(f"Conversation ID: {data.get('conversation_id')}")
    print(f"\nMessage: {data.get('message')}")
    print(f"\nReasoning: {(data.get('reasoning') or 'None')[:500]}")
    print(f"\nChanges: {len(data.get('resume_changes', []))}")
    for change in data.get('resume_changes', [])[:3]:
        print(f"  - {change.get('section')}: {change.get('change_type')}")
    
    conv_id = data.get('conversation_id')
    
    # Test translation
    print("\n=== TRANSLATE TO SPANISH ===")
    response = requests.post(
        f'{BASE_URL}/chat/message',
        json={
            'message': 'Translate this resume to Spanish for Mexico',
            'resume_id': resume_id,
            'conversation_id': conv_id
        },
        timeout=120
    )
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Agent: {data.get('agent_type')}")
    print(f"\nMessage: {data.get('message')[:500] if data.get('message') else 'No message'}")

if __name__ == "__main__":
    test_full_flow()

