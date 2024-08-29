import hashlib

# Example user data
users = {
    'ricky': hashlib.sha256('llm2024'.encode()).hexdigest(),
    'david': hashlib.sha256('llm2024'.encode()).hexdigest()
}

def check_password(username, password):
    """Check if the provided username and password are correct."""
    if username in users:
        return users[username] == hashlib.sha256(password.encode()).hexdigest()
    return False
