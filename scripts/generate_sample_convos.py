import os
import json

def generate_samples():
    """Generate sample conversations programmatically."""
    print("Generating 10 sample conversations...")
    
    os.makedirs("sample_conversations", exist_ok=True)
    
    topics = [
        "The Universal Machine",
        "Cryptography at Bletchley",
        "Artificial Intelligence",
        "Morphogenesis",
        "Personal Life and Running",
        "The ACE Computer",
        "The Imitation Game",
        "Mathematics and Logic",
        "Wartime Experiences",
        "Future of Computing"
    ]
    
    for i, topic in enumerate(topics, 1):
        filename = f"sample_conversations/conversation_{i:02d}.md"
        content = f"# Sample Conversation: {topic}\n\n"
        content += f"**User:** Can you tell me about your work on {topic.lower()}?\n\n"
        content += f"**Turing:** Indeed, it is a rather fascinating subject. [Generated response placeholder]\n"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            
    print(f"Generated {len(topics)} sample conversations in 'sample_conversations/'.")

if __name__ == "__main__":
    generate_samples()
