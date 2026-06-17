import requests
import time

def test_connection():
    print("Testing Internet Connection...")
    try:
        start = time.time()
        response = requests.get("https://www.google.com", timeout=5)
        duration = time.time() - start
        
        print(f"Status Code: {response.status_code}")
        print(f"Latency: {duration:.2f}s")
        
        if response.status_code == 200:
            print("✅ Internet Connection Successful")
            return True
        else:
            print("⚠️ Connection returned non-200 status")
            return False
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
