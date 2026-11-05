import json
import os
import time
import urllib.request
import urllib.error

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def request(method, endpoint, data=None):
    url = BASE_URL + endpoint
    body = json.dumps(data).encode("utf-8") if data is not None else None

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


# 1. Health check
print("\nTEST 1 - HEALTH")
status, data = request("GET", "/health")
assert status == 200, f"Health failed: {status}"
print("PASS - API opérationnelle")


# 2. Different batch sizes
print("\nTEST 2 - BATCH SIZES")

for size in [1, 5, 50]:
    comments = [f"This is test comment number {i}" for i in range(size)]

    start = time.perf_counter()
    status, data = request("POST", "/predict_batch", {"comments": comments})
    elapsed = (time.perf_counter() - start) * 1000

    assert status == 200, f"Batch {size} failed: {status}"
    assert len(data["predictions"]) == size

    print(f"PASS - Batch {size}: {elapsed:.2f} ms")


# 3. Edge cases
print("\nTEST 3 - EDGE CASES")

edge_cases = [
    "Good!",
    "I hate this video.",
    "😊😍🔥",
    "C'est vraiment excellent!",
    "This video est vraiment good.",
    "A" * 5000
]

status, data = request(
    "POST",
    "/predict_batch",
    {"comments": edge_cases}
)

assert status == 200
assert len(data["predictions"]) == len(edge_cases)

print("PASS - Textes courts, longs, emojis et langues mélangées")


# 4. Invalid requests
print("\nTEST 4 - ERROR HANDLING")

status, _ = request("POST", "/predict_batch", {"comments": []})
assert status == 422, f"Expected 422, got {status}"
print("PASS - Batch vide rejeté")

status, _ = request("POST", "/predict_batch", {})
assert status == 422, f"Expected 422, got {status}"
print("PASS - Données manquantes rejetées")


print("\nTOUS LES TESTS API SONT PASSES.")